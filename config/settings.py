"""
Instagram Network Analysis - Settings Configuration

This module contains all configurable parameters for the Instagram Network Analysis tool.
"""

# Instagram API and Request Settings
INSTAGRAM = {
    # Authentication settings
    "AUTH": {
        "USE_SESSION_COOKIE": True,  # Use session cookie instead of username/password
        "SESSION_COOKIE_NAME": "sessionid",  # Name of the session cookie
        "HEADERS": {  # Default headers to mimic browser requests
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        },
    },
    
    # Rate limiting settings
    "RATE_LIMITS": {
        "MAX_REQUESTS_PER_HOUR": 150,  # Conservative limit to avoid blocks
        "REQUESTS_WINDOW_RESET": 3600,  # Reset window in seconds (1 hour)
        "MIN_REQUEST_INTERVAL": 2.0,  # Minimum seconds between requests
        "MAX_REQUEST_INTERVAL": 5.0,  # Maximum seconds between requests
        "BACKOFF_FACTOR": 1.5,  # Multiplicative factor for exponential backoff
        "JITTER": 0.25,  # Random jitter factor to add to delays (0.0-1.0)
    },
    
    # Retry settings
    "RETRY": {
        "MAX_RETRIES": 5,  # Maximum number of retries per request
        "RETRY_DELAY_BASE": 5,  # Base delay in seconds before retrying
        "RETRY_MAX_DELAY": 300,  # Maximum delay in seconds before retrying
        "RETRY_JITTER": 0.1,  # Random jitter factor for retry delays
    },
    
    # Collection settings
    "COLLECTION": {
        "FOLLOWERS_BATCH_SIZE": 50,  # Number of followers to fetch per request
        "FOLLOWING_BATCH_SIZE": 50,  # Number of following to fetch per request
        "MAX_FOLLOWERS_PER_RUN": 20000,  # Maximum followers to collect in one run
        "MAX_FOLLOWING_PER_USER": 1000,  # Maximum following to collect per user
        "PARALLEL_REQUESTS": 3,  # Number of parallel requests for following collection
        "CHECKPOINT_INTERVAL": 100,  # Save checkpoint after every N users
    },
}

# Data Storage Settings
STORAGE = {
    "LOCAL": {
        "RAW_DATA_DIR": "data/raw",
        "PROCESSED_DATA_DIR": "data/processed",
        "RESULTS_DIR": "data/results",
        "CHECKPOINT_DIR": "data/processed/checkpoints",
        "USE_COMPRESSION": True,  # Whether to compress data files
        "COMPRESSION_LEVEL": 9,  # Compression level (1-9, higher = more compression)
    },
    
    "GCS": {
        "ENABLED": False,  # Whether to use Google Cloud Storage
        "BUCKET_NAME": "",  # GCS bucket name
        "PROJECT_ID": "",  # GCS project ID
        "CREDENTIALS_FILE": "",  # Path to GCS credentials file
        "RAW_DATA_PREFIX": "raw/",
        "PROCESSED_DATA_PREFIX": "processed/",
        "RESULTS_PREFIX": "results/",
    },
}

# Processing Settings
PROCESSING = {
    "CHUNK_SIZE": 1000,  # Number of records to process at once
    "MIN_FOLLOWERS_THRESHOLD": 5,  # Minimum followers to be included in analysis
    "INFLUENCE_SCORE_WEIGHTS": {
        "FOLLOWER_COUNT": 0.4,
        "ENGAGEMENT_RATE": 0.3,
        "VERIFIED_STATUS": 0.2,
        "RECENCY": 0.1,
    },
    "OUTLIER_DETECTION": {
        "ENABLED": True,
        "Z_SCORE_THRESHOLD": 3.0,  # Z-score threshold for outlier detection
    },
}

# Visualization Settings
VISUALIZATION = {
    "DEFAULT_PLOT_SIZE": (12, 8),  # Default plot size in inches
    "MAX_NODES_IN_GRAPH": 100,  # Maximum number of nodes to show in network graph
    "COLOR_PALETTE": "viridis",  # Default color palette
    "EXPORT_FORMATS": ["png", "svg", "pdf"],  # Export formats for visualizations
    "DPI": 300,  # DPI for exported visualizations
}

# Logging Settings
LOGGING = {
    "LEVEL": "INFO",  # Logging level
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "DATE_FORMAT": "%Y-%m-%d %H:%M:%S",
    "FILE": "instagram_network_analysis.log",
}
