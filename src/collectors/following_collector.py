"""
Instagram Network Analysis - Following Collector

This module handles the collection of accounts followed by Instagram users.
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

logger = get_logger('collectors.following_collector')

class FollowingCollector:
    """
    Collects accounts followed by Instagram users with controlled parallelization.
    
    Features:
    - Asynchronous collection with controlled parallelism
    - Local cache to avoid duplicate requests
    - Automatic detection and handling of rate limits
    - Checkpoint system for resumable operations
    """
    
    def __init__(self, auth_session, storage=None, base_dir=None):
        """
        Initialize the following collector.
        
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
        self.following_cache = {}  # Cache of already collected following lists
        self.block_detected = False
        self.semaphore = None  # Will be initialized during collection
    
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
                        
                        # Check if we should pause due to blocks
                        if not self.block_detected:
                            self.block_detected = True
                            logger.warning("Block detected, pausing collection")
                            
                            # Sleep for a longer time to recover
                            await asyncio.sleep(300)  # 5 minutes
                            
                            self.block_detected = False
                            logger.info("Resuming collection after pause")
                        
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
    
    async def _fetch_following_batch(self, user_id, end_cursor=None):
        """
        Fetch a batch of accounts followed by a user.
        
        Args:
            user_id (str): User ID to fetch following for
            end_cursor (str, optional): Pagination cursor
            
        Returns:
            tuple: (following_list, has_next_page, end_cursor)
        """
        # GraphQL query hash for following
        query_hash = "d04b0a864b4b54837c0d870b0e77e076"  # This may change over time
        
        # Query variables
        variables = {
            "id": user_id,
            "include_reel": False,
            "fetch_mutual": False,
            "first": self.settings['FOLLOWING_BATCH_SIZE']
        }
        
        # Add after parameter if we have a cursor
        if end_cursor:
            variables["after"] = end_cursor
        
        # Make request
        data = await self._make_graphql_request(query_hash, variables)
        
        if not data or "data" not in data:
            logger.error(f"Failed to fetch following batch for user {user_id}")
            return [], False, None
        
        try:
            # Extract following data
            user_data = data["data"]["user"]
            edge_follow = user_data["edge_follow"]
            page_info = edge_follow["page_info"]
            
            # Extract following
            following_batch = []
            for edge in edge_follow["edges"]:
                node = edge["node"]
                following = {
                    "id": node["id"],
                    "username": node["username"],
                    "full_name": node["full_name"],
                    "is_private": node["is_private"],
                    "is_verified": node["is_verified"],
                    "profile_pic_url": node["profile_pic_url"],
                    "collected_at": datetime.now().isoformat()
                }
                following_batch.append(following)
            
            # Extract pagination info
            has_next_page = page_info["has_next_page"]
            end_cursor = page_info["end_cursor"] if has_next_page else None
            
            logger.info(f"Fetched {len(following_batch)} following for user {user_id}")
            return following_batch, has_next_page, end_cursor
            
        except KeyError as e:
            logger.error(f"Error parsing following data: {e}")
            return [], False, None
    
    async def collect_following_for_user(self, user_id, username=None, max_following=None):
        """
        Collect accounts followed by a single user.
        
        Args:
            user_id (str): User ID to collect following for
            username (str, optional): Username for logging purposes
            max_following (int, optional): Maximum number of following to collect
            
        Returns:
            list: Collected following accounts
        """
        # Check cache first
        if user_id in self.following_cache:
            logger.info(f"Using cached following for user {user_id}")
            return self.following_cache[user_id]
        
        # Set defaults
        if max_following is None:
            max_following = self.settings['MAX_FOLLOWING_PER_USER']
        
        # Initialize
        following = []
        has_next_page = True
        end_cursor = None
        collected_count = 0
        
        user_display = username or user_id
        logger.info(f"Starting following collection for {user_display} (max: {max_following})")
        
        # Collect following
        while has_next_page and collected_count < max_following:
            # Fetch batch
            batch, has_next_page, end_cursor = await self._fetch_following_batch(user_id, end_cursor)
            
            if not batch:
                logger.warning(f"Empty batch received for {user_display}, stopping collection")
                break
            
            # Add to following list
            following.extend(batch)
            collected_count += len(batch)
            
            logger.info(f"Collected {collected_count}/{max_following} following for {user_display}")
            
            # Add random delay between batches
            delay = random.uniform(1.0, 2.0)
            await asyncio.sleep(delay)
            
            # Check if we should stop due to blocks
            if self.block_detected:
                logger.warning(f"Stopping collection for {user_display} due to block detection")
                break
        
        # Cache the results
        self.following_cache[user_id] = following
        
        logger.info(f"Following collection complete for {user_display}. Total collected: {len(following)}")
        return following
    
    async def collect_following_for_followers(self, followers, max_following=None, max_parallel=None):
        """
        Collect accounts followed by a list of followers with controlled parallelism.
        
        Args:
            followers (list): List of follower dictionaries with 'id' and 'username' keys
            max_following (int, optional): Maximum number of following to collect per user
            max_parallel (int, optional): Maximum number of parallel requests
            
        Returns:
            dict: Dictionary mapping user IDs to their following lists
        """
        # Set defaults
        if max_following is None:
            max_following = self.settings['MAX_FOLLOWING_PER_USER']
        
        if max_parallel is None:
            max_parallel = self.settings['PARALLEL_REQUESTS']
        
        # Initialize semaphore for controlled parallelism
        self.semaphore = asyncio.Semaphore(max_parallel)
        
        # Initialize results
        results = {}
        total_followers = len(followers)
        
        logger.info(f"Starting following collection for {total_followers} followers (max parallel: {max_parallel})")
        
        # Create tasks
        async def collect_with_semaphore(follower):
            async with self.semaphore:
                user_id = follower['id']
                username = follower.get('username', user_id)
                
                try:
                    following = await self.collect_following_for_user(
                        user_id=user_id,
                        username=username,
                        max_following=max_following
                    )
                    
                    return user_id, following
                except Exception as e:
                    logger.error(f"Error collecting following for {username}: {e}")
                    return user_id, []
        
        # Process in batches to avoid creating too many tasks at once
        batch_size = 100
        for i in range(0, total_followers, batch_size):
            batch = followers[i:i+batch_size]
            
            # Create tasks for batch
            tasks = [collect_with_semaphore(follower) for follower in batch]
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks)
            
            # Process results
            for user_id, following in batch_results:
                results[user_id] = following
            
            # Save checkpoint
            self._save_checkpoint(results)
            
            logger.info(f"Processed {min(i+batch_size, total_followers)}/{total_followers} followers")
            
            # Add delay between batches
            await asyncio.sleep(5)
        
        # Save final results
        self._save_results(results)
        
        logger.info(f"Following collection complete for all followers. Total processed: {len(results)}")
        return results
    
    def _save_checkpoint(self, results):
        """
        Save a checkpoint of the current collection state.
        
        Args:
            results (dict): Current results
            
        Returns:
            str: Path to checkpoint file
        """
        checkpoint_data = {
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path = self.storage.save_checkpoint(
            data=checkpoint_data,
            checkpoint_name="following_collection"
        )
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def _save_results(self, results):
        """
        Save the final results of the collection.
        
        Args:
            results (dict): Collection results
            
        Returns:
            str: Path to results file
        """
        # Save as JSON
        filename = f"following_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_path = self.storage.save_data(
            data=results,
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
            dict: Checkpoint data
        """
        try:
            # If no specific checkpoint is provided, load the latest one
            if not checkpoint_path:
                checkpoint_data, checkpoint_path = self.storage.load_latest_checkpoint(
                    checkpoint_prefix="following_collection"
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
                return {}
            
            # Restore cache from results
            self.following_cache = checkpoint_data["results"]
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"Restored {len(self.following_cache)} cached following lists")
            
            return checkpoint_data["results"]
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {}
