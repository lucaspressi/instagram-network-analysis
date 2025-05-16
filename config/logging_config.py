"""
Instagram Network Analysis - Logging Configuration

This module configures the logging system for the Instagram Network Analysis tool.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

from config.settings import LOGGING

def setup_logging(log_dir=None, log_file=None):
    """
    Configure the logging system for the application.
    
    Args:
        log_dir (str, optional): Directory to store log files. If None, uses current directory.
        log_file (str, optional): Name of the log file. If None, uses the name from settings.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('instagram_network_analysis')
    logger.setLevel(getattr(logging, LOGGING['LEVEL']))
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=LOGGING['FORMAT'],
        datefmt=LOGGING['DATE_FORMAT']
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOGGING['LEVEL']))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        log_filename = log_file or LOGGING['FILE']
        # Add timestamp to log filename to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename_parts = log_filename.split('.')
        if len(log_filename_parts) > 1:
            log_filename = f"{log_filename_parts[0]}_{timestamp}.{log_filename_parts[1]}"
        else:
            log_filename = f"{log_filename}_{timestamp}.log"
        
        file_path = log_dir_path / log_filename
        
        # Create rotating file handler (10 MB max size, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, LOGGING['LEVEL']))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    logger.info("Logging system initialized")
    return logger

def get_logger(name=None):
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str, optional): Name of the logger. If None, returns the root logger.
    
    Returns:
        logging.Logger: Logger instance
    """
    if name:
        return logging.getLogger(f'instagram_network_analysis.{name}')
    return logging.getLogger('instagram_network_analysis')
