#!/usr/bin/env python3
"""
Step 2.5 Configuration and Test Scripts
"""

# config.py - Configuration for DEM acquisition
class DEMConfig:
    """Configuration settings for DEM acquisition."""
    
    # Download settings
    MAX_CONCURRENT_DOWNLOADS = 8  # Concurrent downloads for speed
    REQUEST_TIMEOUT_SECONDS = 30   # HTTP request timeout
    MAX_RETRIES = 3               # Retry attempts per tile
    BUFFER_KM = 2.0               # Buffer around AOIs in kilometers
    
    # File paths
    STEP2_OUTPUTS_DIR = None      # Auto-detect if None
    OUTPUT_DIR = "outputs/aoi_specific_dems"
    LOGS_DIR = "logs"
    
    # Primary SRTM source (fastest)
    PRIMARY_SRTM_BASE_URL = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1"
    
    # Fallback sources
    FALLBACK_URLS = [
        "https://dds.cr.usgs.gov/srtm/version2_1/SRTM3",
        "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11"
    ]
