"""
Instagram Network Analysis - Authentication Utilities

This module handles authentication with Instagram using session cookies.
"""

import json
import time
import logging
import requests
from pathlib import Path

from config.settings import INSTAGRAM
from config.logging_config import get_logger

logger = get_logger('utils.auth')

class InstagramAuth:
    """
    Handles authentication with Instagram using session cookies.
    """
    
    def __init__(self, session_cookie=None, cookie_file=None):
        """
        Initialize the authentication handler.
        
        Args:
            session_cookie (str, optional): Session cookie value
            cookie_file (str, optional): Path to file containing session cookie
        """
        self.session = requests.Session()
        self.headers = INSTAGRAM['AUTH']['HEADERS']
        self.session.headers.update(self.headers)
        self.authenticated = False
        
        if session_cookie:
            self.set_session_cookie(session_cookie)
        elif cookie_file:
            self.load_cookie_from_file(cookie_file)
    
    def set_session_cookie(self, session_cookie):
        """
        Set the session cookie for authentication.
        
        Args:
            session_cookie (str): Session cookie value
            
        Returns:
            bool: True if cookie was set successfully
        """
        cookie_name = INSTAGRAM['AUTH']['SESSION_COOKIE_NAME']
        self.session.cookies.set(cookie_name, session_cookie, domain='.instagram.com')
        logger.info("Session cookie set")
        return self.validate_auth()
    
    def load_cookie_from_file(self, cookie_file):
        """
        Load session cookie from a file.
        
        Args:
            cookie_file (str): Path to file containing session cookie
            
        Returns:
            bool: True if cookie was loaded and set successfully
        """
        try:
            cookie_path = Path(cookie_file)
            if not cookie_path.exists():
                logger.error(f"Cookie file not found: {cookie_file}")
                return False
            
            with open(cookie_path, 'r') as f:
                cookie_data = json.load(f)
            
            if isinstance(cookie_data, dict) and 'sessionid' in cookie_data:
                session_cookie = cookie_data['sessionid']
            elif isinstance(cookie_data, str):
                session_cookie = cookie_data
            else:
                logger.error("Invalid cookie file format")
                return False
            
            return self.set_session_cookie(session_cookie)
        
        except Exception as e:
            logger.error(f"Error loading cookie from file: {e}")
            return False
    
    def validate_auth(self):
        """
        Validate that the authentication is working by making a test request.
        
        Returns:
            bool: True if authentication is valid
        """
        try:
            # Make a request to a profile page that requires authentication
            response = self.session.get('https://www.instagram.com/accounts/edit/', 
                                        allow_redirects=False)
            
            # If we get redirected to login page, authentication failed
            if response.status_code == 302 or 'login' in response.url:
                logger.error("Authentication validation failed: Redirected to login page")
                self.authenticated = False
                return False
            
            # Check if we got a successful response
            if response.status_code == 200:
                logger.info("Authentication validation successful")
                self.authenticated = True
                return True
            
            logger.error(f"Authentication validation failed: Status code {response.status_code}")
            self.authenticated = False
            return False
            
        except Exception as e:
            logger.error(f"Error validating authentication: {e}")
            self.authenticated = False
            return False
    
    def get_session(self):
        """
        Get the authenticated session.
        
        Returns:
            requests.Session: Authenticated session
        """
        if not self.authenticated:
            logger.warning("Returning unauthenticated session")
        
        return self.session
    
    def save_cookie_to_file(self, cookie_file):
        """
        Save the current session cookie to a file.
        
        Args:
            cookie_file (str): Path to save the cookie file
            
        Returns:
            bool: True if cookie was saved successfully
        """
        try:
            cookie_path = Path(cookie_file)
            cookie_path.parent.mkdir(parents=True, exist_ok=True)
            
            cookie_name = INSTAGRAM['AUTH']['SESSION_COOKIE_NAME']
            session_cookie = self.session.cookies.get(cookie_name, domain='.instagram.com')
            
            if not session_cookie:
                logger.error("No session cookie found to save")
                return False
            
            with open(cookie_path, 'w') as f:
                json.dump({'sessionid': session_cookie}, f)
            
            logger.info(f"Session cookie saved to {cookie_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cookie to file: {e}")
            return False
