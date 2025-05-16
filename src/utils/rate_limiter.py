"""
Instagram Network Analysis - Rate Limiting Utilities

This module handles rate limiting for Instagram API requests to avoid blocks.
"""

import time
import random
import logging
from collections import deque
from datetime import datetime, timedelta

from config.settings import INSTAGRAM
from config.logging_config import get_logger

logger = get_logger('utils.rate_limiter')

class RateLimiter:
    """
    Implements adaptive rate limiting for Instagram API requests.
    
    Features:
    - Tracks request history to enforce hourly limits
    - Implements exponential backoff for failed requests
    - Adds random jitter to delays to avoid detection patterns
    - Adapts delay times based on response patterns
    """
    
    def __init__(self):
        """Initialize the rate limiter with settings from config."""
        self.settings = INSTAGRAM['RATE_LIMITS']
        self.request_history = deque(maxlen=self.settings['MAX_REQUESTS_PER_HOUR'])
        self.current_delay = self.settings['MIN_REQUEST_INTERVAL']
        self.consecutive_errors = 0
        self.last_request_time = 0
    
    def wait(self):
        """
        Wait the appropriate amount of time before making the next request.
        
        Returns:
            float: The actual wait time in seconds
        """
        now = time.time()
        
        # Calculate time since last request
        time_since_last = now - self.last_request_time if self.last_request_time > 0 else float('inf')
        
        # Check if we need to wait based on current delay
        if time_since_last < self.current_delay:
            wait_time = self.current_delay - time_since_last
            
            # Add jitter to avoid detection patterns
            jitter = random.uniform(-self.settings['JITTER'], self.settings['JITTER'])
            jitter_factor = 1.0 + jitter
            wait_time = max(0.1, wait_time * jitter_factor)
            
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before next request")
            time.sleep(wait_time)
            actual_wait = wait_time
        else:
            actual_wait = 0
        
        # Check if we're approaching hourly limit
        self._enforce_hourly_limit()
        
        # Update last request time
        self.last_request_time = time.time()
        
        return actual_wait
    
    def _enforce_hourly_limit(self):
        """
        Enforce the hourly request limit by waiting if necessary.
        """
        # If we haven't reached the max requests per hour, no need to check
        if len(self.request_history) < self.settings['MAX_REQUESTS_PER_HOUR']:
            return
        
        # Get the timestamp of the oldest request in our history
        oldest_request = self.request_history[0]
        now = datetime.now()
        
        # Calculate how long ago the oldest request was made
        time_diff = now - oldest_request
        
        # If the oldest request was less than an hour ago, we need to wait
        if time_diff < timedelta(seconds=self.settings['REQUESTS_WINDOW_RESET']):
            # Calculate how long to wait until we can make another request
            wait_seconds = self.settings['REQUESTS_WINDOW_RESET'] - time_diff.total_seconds()
            
            logger.warning(f"Hourly limit reached. Waiting {wait_seconds:.2f}s before next request")
            time.sleep(wait_seconds)
    
    def record_request(self, success=True):
        """
        Record a request in the history and adjust delay based on success.
        
        Args:
            success (bool): Whether the request was successful
            
        Returns:
            float: The current delay for the next request
        """
        # Record the request timestamp
        self.request_history.append(datetime.now())
        
        # Adjust delay based on success/failure
        if success:
            # Successful request, gradually decrease delay
            self.consecutive_errors = 0
            self.current_delay = max(
                self.settings['MIN_REQUEST_INTERVAL'],
                self.current_delay / 1.2
            )
        else:
            # Failed request, increase delay with exponential backoff
            self.consecutive_errors += 1
            backoff_factor = self.settings['BACKOFF_FACTOR'] ** self.consecutive_errors
            self.current_delay = min(
                self.settings['MAX_REQUEST_INTERVAL'] * backoff_factor,
                300  # Cap at 5 minutes max delay
            )
            logger.warning(f"Request failed. New delay: {self.current_delay:.2f}s (consecutive errors: {self.consecutive_errors})")
        
        return self.current_delay
    
    def detect_rate_limit(self, response):
        """
        Detect if a response indicates rate limiting and adjust accordingly.
        
        Args:
            response (requests.Response): The response to check
            
        Returns:
            bool: True if rate limited, False otherwise
        """
        # Check for rate limit response codes
        if response.status_code in (429, 403):
            logger.warning(f"Rate limit detected (status code: {response.status_code})")
            
            # Increase delay significantly
            self.consecutive_errors += 2  # Count as multiple errors
            backoff_factor = self.settings['BACKOFF_FACTOR'] ** self.consecutive_errors
            self.current_delay = min(
                self.settings['MAX_REQUEST_INTERVAL'] * backoff_factor,
                600  # Cap at 10 minutes max delay for rate limits
            )
            
            # If there's a Retry-After header, use that
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    retry_seconds = int(retry_after)
                    logger.info(f"Respecting Retry-After header: {retry_seconds}s")
                    time.sleep(retry_seconds)
                except (ValueError, TypeError):
                    pass
            
            return True
        
        # Check response body for rate limit indicators
        if response.text and any(x in response.text.lower() for x in 
                                ['rate limit', 'too many requests', 'try again later']):
            logger.warning("Rate limit detected in response body")
            
            # Increase delay significantly
            self.consecutive_errors += 1
            backoff_factor = self.settings['BACKOFF_FACTOR'] ** self.consecutive_errors
            self.current_delay = min(
                self.settings['MAX_REQUEST_INTERVAL'] * backoff_factor,
                300  # Cap at 5 minutes max delay
            )
            
            return True
        
        return False
