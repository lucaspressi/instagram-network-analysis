"""
Instagram Network Analysis - Network Processor

This module processes collected follower and following data to generate network insights.
"""

import json
import logging
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path

from config.settings import PROCESSING
from config.logging_config import get_logger
from src.utils.storage import DataStorage

logger = get_logger('processors.network_processor')

class NetworkProcessor:
    """
    Processes Instagram network data to generate insights and rankings.
    
    Features:
    - Streaming aggregation for large datasets
    - Advanced metrics calculation
    - Outlier detection and filtering
    - Cluster identification
    """
    
    def __init__(self, storage=None, base_dir=None):
        """
        Initialize the network processor.
        
        Args:
            storage (DataStorage, optional): Storage handler
            base_dir (str, optional): Base directory for storage
        """
        # Initialize storage
        if storage:
            self.storage = storage
        else:
            self.storage = DataStorage(base_dir=base_dir)
        
        # Processing settings
        self.settings = PROCESSING
        
        # State variables
        self.target_username = None
        self.followers = []
        self.following_data = {}
        self.account_metrics = {}
        self.rankings = {}
    
    def load_data(self, followers_file, following_file):
        """
        Load follower and following data from files.
        
        Args:
            followers_file (str): Path to followers data file
            following_file (str): Path to following data file
            
        Returns:
            bool: True if data was loaded successfully
        """
        try:
            # Load followers
            self.followers = self.storage.load_data(
                filename=Path(followers_file).name,
                data_type='raw',
                format='json',
                decompress=True if followers_file.endswith('.gz') else False
            )
            
            if not self.followers:
                logger.error(f"Failed to load followers data from {followers_file}")
                return False
            
            # Load following data
            self.following_data = self.storage.load_data(
                filename=Path(following_file).name,
                data_type='raw',
                format='json',
                decompress=True if following_file.endswith('.gz') else False
            )
            
            if not self.following_data:
                logger.error(f"Failed to load following data from {following_file}")
                return False
            
            # Extract target username from followers file
            if followers_file.startswith('followers_'):
                parts = followers_file.split('_')
                if len(parts) > 1:
                    self.target_username = parts[1]
            
            logger.info(f"Loaded {len(self.followers)} followers and following data for {len(self.following_data)} users")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def process_network_data(self):
        """
        Process network data to generate insights.
        
        Returns:
            dict: Processing results
        """
        logger.info("Starting network data processing")
        
        # Convert following data to a flat list of all accounts followed
        all_following = self._flatten_following_data()
        
        # Calculate account frequencies
        account_counts = self._calculate_account_frequencies(all_following)
        
        # Calculate additional metrics
        self._calculate_account_metrics(account_counts)
        
        # Generate rankings
        self._generate_rankings()
        
        # Save results
        self._save_results()
        
        logger.info("Network data processing complete")
        return self.rankings
    
    def _flatten_following_data(self):
        """
        Convert following data to a flat list of all accounts followed.
        
        Returns:
            list: Flat list of all accounts followed
        """
        all_following = []
        
        # Process in chunks to handle large datasets
        chunk_size = self.settings['CHUNK_SIZE']
        follower_ids = list(self.following_data.keys())
        
        for i in range(0, len(follower_ids), chunk_size):
            chunk = follower_ids[i:i+chunk_size]
            
            for follower_id in chunk:
                following_list = self.following_data.get(follower_id, [])
                all_following.extend(following_list)
            
            logger.info(f"Processed {min(i+chunk_size, len(follower_ids))}/{len(follower_ids)} followers' following data")
        
        logger.info(f"Flattened following data: {len(all_following)} total accounts followed")
        return all_following
    
    def _calculate_account_frequencies(self, all_following):
        """
        Calculate frequency of each account in the following data.
        
        Args:
            all_following (list): Flat list of all accounts followed
            
        Returns:
            dict: Dictionary mapping account IDs to their frequency
        """
        # Extract account IDs
        account_ids = [account['id'] for account in all_following]
        
        # Count frequencies
        account_counts = Counter(account_ids)
        
        logger.info(f"Calculated frequencies for {len(account_counts)} unique accounts")
        return account_counts
    
    def _calculate_account_metrics(self, account_counts):
        """
        Calculate additional metrics for each account.
        
        Args:
            account_counts (Counter): Dictionary mapping account IDs to their frequency
        """
        # Create a mapping of account ID to account data
        account_data = {}
        for account in self._flatten_following_data():
            account_id = account['id']
            if account_id not in account_data:
                account_data[account_id] = account
        
        # Calculate metrics for each account
        total_followers = len(self.following_data)
        
        for account_id, count in account_counts.items():
            # Skip accounts with too few followers
            if count < self.settings['MIN_FOLLOWERS_THRESHOLD']:
                continue
            
            # Get account data
            account = account_data.get(account_id, {})
            
            # Calculate penetration rate (percentage of target's followers who follow this account)
            penetration_rate = (count / total_followers) * 100 if total_followers > 0 else 0
            
            # Calculate influence score
            influence_score = self._calculate_influence_score(account, count, total_followers)
            
            # Store metrics
            self.account_metrics[account_id] = {
                'id': account_id,
                'username': account.get('username', 'unknown'),
                'full_name': account.get('full_name', ''),
                'is_verified': account.get('is_verified', False),
                'follower_count': count,
                'penetration_rate': penetration_rate,
                'influence_score': influence_score
            }
        
        logger.info(f"Calculated metrics for {len(self.account_metrics)} accounts")
    
    def _calculate_influence_score(self, account, count, total_followers):
        """
        Calculate influence score for an account.
        
        Args:
            account (dict): Account data
            count (int): Number of followers who follow this account
            total_followers (int): Total number of followers
            
        Returns:
            float: Influence score
        """
        weights = self.settings['INFLUENCE_SCORE_WEIGHTS']
        
        # Follower count component (normalized)
        follower_component = min(count / total_followers * 10, 1.0) * weights['FOLLOWER_COUNT']
        
        # Verified status component
        verified_component = 1.0 if account.get('is_verified', False) else 0.0
        verified_component *= weights['VERIFIED_STATUS']
        
        # Engagement rate component (placeholder - would need actual engagement data)
        engagement_component = 0.5 * weights['ENGAGEMENT_RATE']
        
        # Recency component (placeholder - would need posting frequency data)
        recency_component = 0.5 * weights['RECENCY']
        
        # Calculate total score
        influence_score = follower_component + verified_component + engagement_component + recency_component
        
        return influence_score
    
    def _detect_outliers(self, data, column):
        """
        Detect outliers in data using Z-score method.
        
        Args:
            data (pd.DataFrame): Data to detect outliers in
            column (str): Column to detect outliers in
            
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        if not self.settings['OUTLIER_DETECTION']['ENABLED']:
            return data
        
        # Calculate Z-scores
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        
        # Filter out outliers
        threshold = self.settings['OUTLIER_DETECTION']['Z_SCORE_THRESHOLD']
        filtered_data = data[z_scores < threshold]
        
        logger.info(f"Outlier detection removed {len(data) - len(filtered_data)} accounts")
        return filtered_data
    
    def _generate_rankings(self):
        """
        Generate rankings based on different metrics.
        """
        # Convert metrics to DataFrame for easier processing
        df = pd.DataFrame.from_dict(self.account_metrics, orient='index')
        
        # Apply outlier detection if enabled
        if self.settings['OUTLIER_DETECTION']['ENABLED']:
            df = self._detect_outliers(df, 'follower_count')
        
        # Generate rankings by follower count
        follower_ranking = df.sort_values('follower_count', ascending=False).head(100).to_dict('records')
        
        # Generate rankings by penetration rate
        penetration_ranking = df.sort_values('penetration_rate', ascending=False).head(100).to_dict('records')
        
        # Generate rankings by influence score
        influence_ranking = df.sort_values('influence_score', ascending=False).head(100).to_dict('records')
        
        # Store rankings
        self.rankings = {
            'by_follower_count': follower_ranking,
            'by_penetration_rate': penetration_ranking,
            'by_influence_score': influence_ranking,
            'metadata': {
                'target_username': self.target_username,
                'total_followers_analyzed': len(self.following_data),
                'total_unique_accounts': len(df),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        logger.info("Generated rankings based on follower count, penetration rate, and influence score")
    
    def _save_results(self):
        """
        Save processing results.
        
        Returns:
            str: Path to results file
        """
        # Save as JSON
        filename = f"rankings_{self.target_username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_path = self.storage.save_data(
            data=self.rankings,
            filename=filename,
            data_type='results',
            format='json',
            compress=True
        )
        
        logger.info(f"Results saved: {results_path}")
        return results_path
    
    def identify_clusters(self, n_clusters=5):
        """
        Identify clusters of interest among followers.
        
        Args:
            n_clusters (int): Number of clusters to identify
            
        Returns:
            dict: Cluster information
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            
            logger.info(f"Identifying {n_clusters} interest clusters")
            
            # Convert metrics to DataFrame
            df = pd.DataFrame.from_dict(self.account_metrics, orient='index')
            
            # Use account names as features for clustering
            texts = df['username'].fillna('') + ' ' + df['full_name'].fillna('')
            
            # Create TF-IDF features
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(X)
            
            # Identify top accounts in each cluster
            clusters = {}
            for i in range(n_clusters):
                cluster_df = df[df['cluster'] == i].sort_values('follower_count', ascending=False)
                
                # Get top features (words) for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_features_idx = cluster_center.argsort()[-10:]
                top_features = [vectorizer.get_feature_names_out()[idx] for idx in top_features_idx]
                
                clusters[f"cluster_{i}"] = {
                    'top_accounts': cluster_df.head(10).to_dict('records'),
                    'size': len(cluster_df),
                    'top_features': top_features
                }
            
            # Save cluster results
            filename = f"clusters_{self.target_username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            self.storage.save_data(
                data=clusters,
                filename=filename,
                data_type='results',
                format='json',
                compress=True
            )
            
            logger.info(f"Identified {n_clusters} clusters")
            return clusters
            
        except ImportError:
            logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            return {}
            
        except Exception as e:
            logger.error(f"Error identifying clusters: {e}")
            return {}
