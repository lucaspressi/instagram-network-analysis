"""
Instagram Network Analysis - Follower Collector

This module handles the collection of followers from Instagram profiles.
"""

import json
import time
import asyncio
import logging
import aiohttp
import random
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

from config.settings import INSTAGRAM
from config.logging_config import get_logger
from src.utils.rate_limiter import RateLimiter
from src.utils.storage import DataStorage

logger = get_logger('collectors.follower_collector')

class FollowerCollector:
    """
    Collects followers from Instagram profiles using session-based authentication.
    
    Features:
    - Asynchronous collection for improved performance
    - Checkpoint system for resumable operations
    - Adaptive rate limiting to avoid blocks
    """
    
    def __init__(self, auth_session, storage=None, base_dir=None):
        """
        Initialize the follower collector.
        
        Args:
            auth_session (requests.Session): Authenticated session
            storage (DataStorage, optional): Storage handler
            base_dir (str, optional): Base directory for storage
        """
        self.session = auth_session
        self.rate_limiter = RateLimiter()
        
        # Initialize storage
        if storage:
            self.storage = storage
        else:
            self.storage = DataStorage(base_dir=base_dir)
        
        # Collection settings
        self.settings = INSTAGRAM['COLLECTION']
        
        # State variables
        self.target_username = None
        self.target_user_id = None
        self.followers_count = 0
        self.collected_count = 0
        self.has_next_page = True
        self.end_cursor = None
        self.followers = []
    
    async def _make_graphql_request(self, query_hash, variables):
        """
        Make a GraphQL request to Instagram.
        
        Args:
            query_hash (str): GraphQL query hash
            variables (dict): Query variables
            
        Returns:
            dict: Response data
        """
        # Wait for rate limiter
        self.rate_limiter.wait()
        
        # Encode variables
        variables_json = json.dumps(variables)
        encoded_variables = quote(variables_json)
        
        # Construct URL
        url = f"https://www.instagram.com/graphql/query/?query_hash={query_hash}&variables={encoded_variables}"
        
        # Make request
        try:
            async with aiohttp.ClientSession(cookies=self.session.cookies, 
                                            headers=self.session.headers) as session:
                async with session.get(url) as response:
                    # Check for rate limiting
                    if response.status in (429, 403):
                        logger.warning(f"Rate limited (status: {response.status})")
                        self.rate_limiter.record_request(success=False)
                        return None
                    
                    # Parse response
                    if response.status == 200:
                        data = await response.json()
                        self.rate_limiter.record_request(success=True)
                        return data
                    else:
                        logger.error(f"Request failed with status {response.status}")
                        self.rate_limiter.record_request(success=False)
                        return None
        
        except Exception as e:
            logger.error(f"Error making GraphQL request: {e}")
            self.rate_limiter.record_request(success=False)
            return None
    
    async def _get_user_id(self, username):
        """
        Get user ID from username.
        
        Args:
            username (str): Instagram username
            
        Returns:
            str: User ID
        """
        # Wait for rate limiter
        self.rate_limiter.wait()
        
        # Make request to profile page
        url = f"https://www.instagram.com/{username}/"
        
        try:
            async with aiohttp.ClientSession(cookies=self.session.cookies, 
                                            headers=self.session.headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Extract user ID from HTML
                        import re
                        match = re.search(r'"user_id":"(\d+)"', html)
                        if match:
                            user_id = match.group(1)
                            logger.info(f"Found user ID for {username}: {user_id}")
                            
                            # Also extract follower count
                            count_match = re.search(r'"edge_followed_by":{"count":(\d+)}', html)
                            if count_match:
                                self.followers_count = int(count_match.group(1))
                                logger.info(f"Follower count for {username}: {self.followers_count}")
                            
                            self.rate_limiter.record_request(success=True)
                            return user_id
                    
                    logger.error(f"Failed to get user ID for {username} (status: {response.status})")
                    self.rate_limiter.record_request(success=False)
                    return None
        
        except Exception as e:
            logger.error(f"Error getting user ID: {e}")
            self.rate_limiter.record_request(success=False)
            return None
    
    async def _fetch_followers_batch(self):
        """
        Fetch a batch of followers.
        
        Returns:
            tuple: (followers_list, has_next_page, end_cursor)
        """
        # GraphQL query hash for followers
        query_hash = "c76146de99bb02f6415203be841dd25a"  # This may change over time
        
        # Query variables
        variables = {
            "id": self.target_user_id,
            "include_reel": False,
            "fetch_mutual": False,
            "first": self.settings['FOLLOWERS_BATCH_SIZE']
        }
        
        # Add after parameter if we have a cursor
        if self.end_cursor:
            variables["after"] = self.end_cursor
        
        # Make request
        data = await self._make_graphql_request(query_hash, variables)
        
        if not data or "data" not in data:
            logger.error("Failed to fetch followers batch")
            return [], False, None
        
        try:
            # Extract followers data
            user_data = data["data"]["user"]
            edge_followed_by = user_data["edge_followed_by"]
            page_info = edge_followed_by["page_info"]
            
            # Extract followers
            followers_batch = []
            for edge in edge_followed_by["edges"]:
                node = edge["node"]
                follower = {
                    "id": node["id"],
                    "username": node["username"],
                    "full_name": node["full_name"],
                    "is_private": node["is_private"],
                    "is_verified": node["is_verified"],
                    "profile_pic_url": node["profile_pic_url"],
                    "collected_at": datetime.now().isoformat()
                }
                followers_batch.append(follower)
            
            # Extract pagination info
            has_next_page = page_info["has_next_page"]
            end_cursor = page_info["end_cursor"] if has_next_page else None
            
            logger.info(f"Fetched {len(followers_batch)} followers")
            return followers_batch, has_next_page, end_cursor
            
        except KeyError as e:
            logger.error(f"Error parsing followers data: {e}")
            return [], False, None
    
    async def collect_followers(self, username, max_followers=None, save_interval=None):
        """
        Collect followers for a given username.
        
        Args:
            username (str): Instagram username to collect followers from
            max_followers (int, optional): Maximum number of followers to collect
            save_interval (int, optional): Save checkpoint after every N followers
            
        Returns:
            list: Collected followers
        """
        self.target_username = username
        
        # Set defaults
        if max_followers is None:
            max_followers = self.settings['MAX_FOLLOWERS_PER_RUN']
        
        if save_interval is None:
            save_interval = self.settings['CHECKPOINT_INTERVAL']
        
        # Get user ID
        self.target_user_id = await self._get_user_id(username)
        if not self.target_user_id:
            logger.error(f"Could not get user ID for {username}")
            return []
        
        # Reset state
        self.collected_count = 0
        self.has_next_page = True
        self.end_cursor = None
        self.followers = []
        
        logger.info(f"Starting follower collection for {username} (max: {max_followers})")
        
        # Collect followers
        while self.has_next_page and self.collected_count < max_followers:
            # Fetch batch
            batch, self.has_next_page, self.end_cursor = await self._fetch_followers_batch()
            
            if not batch:
                logger.warning("Empty batch received, stopping collection")
                break
            
            # Add to followers list
            self.followers.extend(batch)
            self.collected_count += len(batch)
            
            logger.info(f"Collected {self.collected_count}/{max_followers} followers")
            
            # Save checkpoint if needed
            if self.collected_count % save_interval == 0:
                self._save_checkpoint()
            
            # Add random delay between batches
            delay = random.uniform(1.0, 3.0)
            await asyncio.sleep(delay)
        
        # Save final results
        self._save_results()
        
        logger.info(f"Follower collection complete. Total collected: {self.collected_count}")
        return self.followers
    
    def _save_checkpoint(self):
        """
        Save a checkpoint of the current collection state.
        
        Returns:
            str: Path to checkpoint file
        """
        checkpoint_data = {
            "target_username": self.target_username,
            "target_user_id": self.target_user_id,
            "followers_count": self.followers_count,
            "collected_count": self.collected_count,
            "has_next_page": self.has_next_page,
            "end_cursor": self.end_cursor,
            "followers": self.followers,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path = self.storage.save_checkpoint(
            data=checkpoint_data,
            checkpoint_name=f"followers_{self.target_username}"
        )
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def _save_results(self):
        """
        Save the final results of the collection.
        
        Returns:
            str: Path to results file
        """
        # Save as JSON
        filename = f"followers_{self.target_username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_path = self.storage.save_data(
            data=self.followers,
            filename=filename,
            data_type='raw',
            format='json',
            compress=True
        )
        
        logger.info(f"Results saved: {results_path}")
        return results_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """
        Load a checkpoint to resume collection.
        
        Args:
            checkpoint_path (str, optional): Path to checkpoint file
            
        Returns:
            bool: True if checkpoint was loaded successfully
        """
        try:
            # If no specific checkpoint is provided, load the latest one
            if not checkpoint_path:
                checkpoint_data, checkpoint_path = self.storage.load_latest_checkpoint(
                    checkpoint_prefix=f"followers_{self.target_username}"
                )
            else:
                checkpoint_data = self.storage.load_data(
                    filename=Path(checkpoint_path).name,
                    data_type='processed',
                    format='pickle',
                    decompress=True
                )
            
            if not checkpoint_data:
                logger.error("No checkpoint data found")
                return False
            
            # Restore state
            self.target_username = checkpoint_data["target_username"]
            self.target_user_id = checkpoint_data["target_user_id"]
            self.followers_count = checkpoint_data["followers_count"]
            self.collected_count = checkpoint_data["collected_count"]
            self.has_next_page = checkpoint_data["has_next_page"]
            self.end_cursor = checkpoint_data["end_cursor"]
            self.followers = checkpoint_data["followers"]
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"Resuming collection for {self.target_username} from {self.collected_count} followers")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
