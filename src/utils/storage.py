"""
Instagram Network Analysis - Data Storage Utilities

This module handles data storage operations, including local and GCS storage.
"""

import os
import json
import gzip
import pickle
import logging
from pathlib import Path
from datetime import datetime

from config.settings import STORAGE
from config.logging_config import get_logger

logger = get_logger('utils.storage')

class DataStorage:
    """
    Handles data storage operations for Instagram Network Analysis.
    
    Features:
    - Local file storage with optional compression
    - Google Cloud Storage integration
    - Checkpoint management for resumable operations
    """
    
    def __init__(self, base_dir=None, use_gcs=None):
        """
        Initialize the data storage handler.
        
        Args:
            base_dir (str, optional): Base directory for local storage
            use_gcs (bool, optional): Whether to use Google Cloud Storage
        """
        self.settings = STORAGE
        
        # Set up local storage
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path('.')
        
        self.raw_dir = self.base_dir / self.settings['LOCAL']['RAW_DATA_DIR']
        self.processed_dir = self.base_dir / self.settings['LOCAL']['PROCESSED_DATA_DIR']
        self.results_dir = self.base_dir / self.settings['LOCAL']['RESULTS_DIR']
        self.checkpoint_dir = self.base_dir / self.settings['LOCAL']['CHECKPOINT_DIR']
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.processed_dir, self.results_dir, self.checkpoint_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Set up GCS if enabled
        self.use_gcs = use_gcs if use_gcs is not None else self.settings['GCS']['ENABLED']
        self.gcs_client = None
        self.gcs_bucket = None
        
        if self.use_gcs:
            self._init_gcs()
    
    def _init_gcs(self):
        """
        Initialize Google Cloud Storage client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            from google.cloud import storage
            from google.oauth2 import service_account
            
            # Set up GCS client
            if self.settings['GCS']['CREDENTIALS_FILE']:
                credentials = service_account.Credentials.from_service_account_file(
                    self.settings['GCS']['CREDENTIALS_FILE']
                )
                self.gcs_client = storage.Client(
                    project=self.settings['GCS']['PROJECT_ID'],
                    credentials=credentials
                )
            else:
                # Use default credentials
                self.gcs_client = storage.Client(project=self.settings['GCS']['PROJECT_ID'])
            
            # Get bucket
            self.gcs_bucket = self.gcs_client.bucket(self.settings['GCS']['BUCKET_NAME'])
            
            logger.info(f"GCS initialized with bucket: {self.settings['GCS']['BUCKET_NAME']}")
            return True
            
        except ImportError:
            logger.error("Google Cloud Storage libraries not installed. Run: pip install google-cloud-storage")
            self.use_gcs = False
            return False
            
        except Exception as e:
            logger.error(f"Error initializing GCS: {e}")
            self.use_gcs = False
            return False
    
    def save_data(self, data, filename, data_type='raw', format='json', compress=None):
        """
        Save data to storage.
        
        Args:
            data: Data to save
            filename (str): Filename to save data to
            data_type (str): Type of data ('raw', 'processed', or 'results')
            format (str): Format to save data in ('json', 'pickle', 'csv')
            compress (bool, optional): Whether to compress the data
            
        Returns:
            str: Path to saved file
        """
        # Determine compression setting
        if compress is None:
            compress = self.settings['LOCAL']['USE_COMPRESSION']
        
        # Determine local directory based on data type
        if data_type == 'raw':
            local_dir = self.raw_dir
            gcs_prefix = self.settings['GCS']['RAW_DATA_PREFIX']
        elif data_type == 'processed':
            local_dir = self.processed_dir
            gcs_prefix = self.settings['GCS']['PROCESSED_DATA_PREFIX']
        elif data_type == 'results':
            local_dir = self.results_dir
            gcs_prefix = self.settings['GCS']['RESULTS_PREFIX']
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        
        # Ensure filename has appropriate extension
        if format == 'json':
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
        elif format == 'pickle':
            if not filename.endswith('.pkl'):
                filename = f"{filename}.pkl"
        elif format == 'csv':
            if not filename.endswith('.csv'):
                filename = f"{filename}.csv"
        
        # Add compression extension if needed
        if compress and not filename.endswith('.gz'):
            filename = f"{filename}.gz"
        
        # Create full local path
        local_path = local_dir / filename
        
        # Save data locally
        try:
            if compress:
                if format == 'json':
                    with gzip.open(local_path, 'wt', encoding='utf-8', 
                                  compresslevel=self.settings['LOCAL']['COMPRESSION_LEVEL']) as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                elif format == 'pickle':
                    with gzip.open(local_path, 'wb',
                                  compresslevel=self.settings['LOCAL']['COMPRESSION_LEVEL']) as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                elif format == 'csv':
                    import pandas as pd
                    if isinstance(data, pd.DataFrame):
                        with gzip.open(local_path, 'wt', encoding='utf-8',
                                      compresslevel=self.settings['LOCAL']['COMPRESSION_LEVEL']) as f:
                            data.to_csv(f, index=False)
                    else:
                        logger.error("Data must be a pandas DataFrame for CSV format")
                        return None
            else:
                if format == 'json':
                    with open(local_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                elif format == 'pickle':
                    with open(local_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                elif format == 'csv':
                    import pandas as pd
                    if isinstance(data, pd.DataFrame):
                        data.to_csv(local_path, index=False)
                    else:
                        logger.error("Data must be a pandas DataFrame for CSV format")
                        return None
            
            logger.info(f"Data saved locally to {local_path}")
            
            # Upload to GCS if enabled
            if self.use_gcs and self.gcs_bucket:
                gcs_path = f"{gcs_prefix}{filename}"
                blob = self.gcs_bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                logger.info(f"Data uploaded to GCS: gs://{self.settings['GCS']['BUCKET_NAME']}/{gcs_path}")
            
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return None
    
    def load_data(self, filename, data_type='raw', format=None, decompress=None):
        """
        Load data from storage.
        
        Args:
            filename (str): Filename to load data from
            data_type (str): Type of data ('raw', 'processed', or 'results')
            format (str, optional): Format of the data ('json', 'pickle', 'csv')
            decompress (bool, optional): Whether to decompress the data
            
        Returns:
            Data loaded from storage
        """
        # Determine local directory based on data type
        if data_type == 'raw':
            local_dir = self.raw_dir
            gcs_prefix = self.settings['GCS']['RAW_DATA_PREFIX']
        elif data_type == 'processed':
            local_dir = self.processed_dir
            gcs_prefix = self.settings['GCS']['PROCESSED_DATA_PREFIX']
        elif data_type == 'results':
            local_dir = self.results_dir
            gcs_prefix = self.settings['GCS']['RESULTS_PREFIX']
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        
        # Create full local path
        local_path = local_dir / filename
        
        # Determine format and compression from filename if not specified
        if format is None:
            if filename.endswith('.json.gz') or filename.endswith('.json'):
                format = 'json'
            elif filename.endswith('.pkl.gz') or filename.endswith('.pkl'):
                format = 'pickle'
            elif filename.endswith('.csv.gz') or filename.endswith('.csv'):
                format = 'csv'
            else:
                logger.error(f"Could not determine format from filename: {filename}")
                return None
        
        if decompress is None:
            decompress = filename.endswith('.gz')
        
        # Check if file exists locally
        if not local_path.exists():
            # Try to download from GCS if enabled
            if self.use_gcs and self.gcs_bucket:
                gcs_path = f"{gcs_prefix}{filename}"
                blob = self.gcs_bucket.blob(gcs_path)
                
                if blob.exists():
                    blob.download_to_filename(local_path)
                    logger.info(f"Downloaded file from GCS: gs://{self.settings['GCS']['BUCKET_NAME']}/{gcs_path}")
                else:
                    logger.error(f"File not found in GCS: gs://{self.settings['GCS']['BUCKET_NAME']}/{gcs_path}")
                    return None
            else:
                logger.error(f"File not found: {local_path}")
                return None
        
        # Load data
        try:
            if decompress:
                if format == 'json':
                    with gzip.open(local_path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                elif format == 'pickle':
                    with gzip.open(local_path, 'rb') as f:
                        data = pickle.load(f)
                elif format == 'csv':
                    import pandas as pd
                    with gzip.open(local_path, 'rt', encoding='utf-8') as f:
                        data = pd.read_csv(f)
            else:
                if format == 'json':
                    with open(local_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif format == 'pickle':
                    with open(local_path, 'rb') as f:
                        data = pickle.load(f)
                elif format == 'csv':
                    import pandas as pd
                    data = pd.read_csv(local_path)
            
            logger.info(f"Data loaded from {local_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def save_checkpoint(self, data, checkpoint_name):
        """
        Save a checkpoint for resumable operations.
        
        Args:
            data: Checkpoint data to save
            checkpoint_name (str): Name of the checkpoint
            
        Returns:
            str: Path to saved checkpoint file
        """
        # Add timestamp to checkpoint name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{checkpoint_name}_{timestamp}.pkl.gz"
        
        return self.save_data(
            data=data,
            filename=filename,
            data_type='processed',
            format='pickle',
            compress=True
        )
    
    def load_latest_checkpoint(self, checkpoint_prefix):
        """
        Load the latest checkpoint for a given prefix.
        
        Args:
            checkpoint_prefix (str): Prefix of the checkpoint name
            
        Returns:
            tuple: (checkpoint_data, checkpoint_path) or (None, None) if no checkpoint found
        """
        # Get all checkpoint files matching the prefix
        checkpoint_files = list(self.checkpoint_dir.glob(f"{checkpoint_prefix}_*.pkl.gz"))
        
        if not checkpoint_files:
            logger.info(f"No checkpoints found for prefix: {checkpoint_prefix}")
            return None, None
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Load the newest checkpoint
        latest_checkpoint = checkpoint_files[0].name
        logger.info(f"Loading latest checkpoint: {latest_checkpoint}")
        
        data = self.load_data(
            filename=latest_checkpoint,
            data_type='processed',
            format='pickle',
            decompress=True
        )
        
        return data, str(checkpoint_files[0])
