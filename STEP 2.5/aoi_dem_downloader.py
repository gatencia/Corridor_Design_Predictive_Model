#!/usr/bin/env python3
"""
AOI-specific DEM Downloader using OpenTopography API.
Downloads and processes DEM data for specific study sites.
"""

import os
import sys
import requests
import rasterio
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AOIDEMDownloader:
    """Downloads DEM data for specific Areas of Interest using OpenTopography API."""
    
    def __init__(self, api_key: str, output_dir: str = "outputs/aoi_specific_dems"):
        """
        Initialize the AOI DEM downloader.
        
        Parameters:
        -----------
        api_key : str
            OpenTopography API key
        output_dir : str
            Directory to save downloaded DEMs
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # FIXED: Use correct endpoint and parameter name based on debugging
        self.base_url = "https://portal.opentopography.org/API/globaldem"
        self.api_key_param = "API_Key"  # Exact capitalization required
        
        # Available DEM types (verified working)
        self.dem_types = {
            'SRTMGL1': {'resolution': 30, 'description': 'SRTM GL1 30m'},
            'SRTMGL3': {'resolution': 90, 'description': 'SRTM GL3 90m'}
        }
        
        self.session = requests.Session()
        
        logger.info(f"Initialized AOI DEM Downloader")
        logger.info(f"Endpoint: {self.base_url}")
        logger.info(f"API key parameter: {self.api_key_param}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def download_dem_for_aoi(self, aoi_name: str, bounds: Tuple[float, float, float, float],
                           dem_type: str = 'SRTMGL1', buffer_degrees: float = 0.01) -> Optional[Path]:
        """
        Download DEM for a specific AOI.
        
        Parameters:
        -----------
        aoi_name : str
            Name for the AOI (used in filename)
        bounds : Tuple[float, float, float, float]
            Bounding box (west, south, east, north) in decimal degrees
        dem_type : str
            DEM type ('SRTMGL1' or 'SRTMGL3')
        buffer_degrees : float
            Buffer to add around AOI in degrees
            
        Returns:
        --------
        Optional[Path]
            Path to downloaded DEM file or None if failed
        """
        west, south, east, north = bounds
        
        # Add buffer
        west -= buffer_degrees
        south -= buffer_degrees
        east += buffer_degrees
        north += buffer_degrees
        
        logger.info(f"Downloading DEM for AOI: {aoi_name}")
        logger.info(f"Bounds: {west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}")
        logger.info(f"DEM type: {dem_type}")
        
        # Validate DEM type
        if dem_type not in self.dem_types:
            logger.error(f"Invalid DEM type: {dem_type}. Available: {list(self.dem_types.keys())}")
            return None
        
        # Build parameters using verified working configuration
        params = {
            'demtype': dem_type,
            'south': str(south),
            'north': str(north),
            'west': str(west),
            'east': str(east),
            'outputFormat': 'GTiff',
            self.api_key_param: self.api_key  # Use correct parameter name
        }
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{aoi_name}_{dem_type}_{timestamp}.tif"
        output_path = self.output_dir / output_filename
        
        try:
            logger.info(f"Making API request to: {self.base_url}")
            logger.debug(f"Parameters: {params}")
            
            # Make request with timeout and retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(
                        self.base_url,
                        params=params,
                        timeout=300,  # 5 minutes timeout
                        stream=True
                    )
                    break
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Request timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            # Check response
            if response.status_code == 200:
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                content_length = response.headers.get('content-length', 'unknown')
                
                logger.info(f"Content-Type: {content_type}")
                logger.info(f"Content-Length: {content_length}")
                
                if 'image/tiff' in content_type or 'application/octet-stream' in content_type:
                    # Valid DEM data
                    logger.info(f"Downloading DEM data to: {output_path}")
                    
                    with open(output_path, 'wb') as f:
                        total_size = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                total_size += len(chunk)
                    
                    logger.info(f"Download completed: {total_size:,} bytes")
                    
                    # Validate downloaded file
                    if self._validate_dem_file(output_path):
                        logger.info(f"✅ Successfully downloaded and validated DEM: {output_path}")
                        return output_path
                    else:
                        logger.error(f"❌ Downloaded file failed validation")
                        output_path.unlink(missing_ok=True)
                        return None
                        
                else:
                    # Likely an error response
                    error_content = response.text[:1000]  # First 1000 chars
                    logger.error(f"❌ Unexpected content type: {content_type}")
                    logger.error(f"Response content: {error_content}")
                    return None
                    
            else:
                error_content = response.text[:1000]
                logger.error(f"❌ HTTP {response.status_code}: {response.reason}")
                logger.error(f"Response: {error_content}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error downloading DEM: {e}")
            return None
    
    def _validate_dem_file(self, dem_path: Path) -> bool:
        """
        Validate downloaded DEM file.
        
        Parameters:
        -----------
        dem_path : Path
            Path to DEM file
            
        Returns:
        --------
        bool
            True if file is valid
        """
        try:
            # Check file size
            file_size = dem_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is likely an error
                logger.error(f"File too small: {file_size} bytes")
                return False
            
            # Try to open with rasterio
            with rasterio.open(dem_path) as src:
                # Check basic properties
                if src.count < 1:
                    logger.error("No bands in raster")
                    return False
                
                if src.width < 1 or src.height < 1:
                    logger.error(f"Invalid dimensions: {src.width}x{src.height}")
                    return False
                
                # Read a small sample
                sample = src.read(1, window=rasterio.windows.Window(0, 0, min(10, src.width), min(10, src.height)))
                if np.all(np.isnan(sample)) or np.all(sample == src.nodata):
                    logger.error("No valid data in sample")
                    return False
                
                logger.info(f"Valid DEM: {src.width}x{src.height}, {src.count} bands")
                return True
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def download_dems_for_study_sites(self, step2_output_dir: str) -> Dict[str, Optional[Path]]:
        """
        Download DEMs for all study sites found in Step 2 outputs.
        
        Parameters:
        -----------
        step2_output_dir : str
            Directory containing Step 2 outputs with study site information
            
        Returns:
        --------
        Dict[str, Optional[Path]]
            Dictionary mapping site names to DEM file paths
        """
        logger.info(f"Scanning Step 2 outputs for study sites: {step2_output_dir}")
        
        step2_dir = Path(step2_output_dir)
        results = {}
        
        # Look for site summary or bounds files
        bounds_files = list(step2_dir.rglob("*bounds*.json")) + list(step2_dir.rglob("*sites*.json"))
        
        if not bounds_files:
            logger.warning("No bounds/sites files found in Step 2 outputs")
            return results
        
        for bounds_file in bounds_files:
            try:
                with open(bounds_file, 'r') as f:
                    site_data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(site_data, dict) and 'sites' in site_data:
                    sites = site_data['sites']
                elif isinstance(site_data, dict):
                    sites = site_data
                elif isinstance(site_data, list):
                    sites = {f"site_{i}": site for i, site in enumerate(site_data)}
                else:
                    continue
                
                for site_name, site_info in sites.items():
                    if 'bounds' in site_info:
                        bounds = site_info['bounds']
                        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
                            logger.info(f"Downloading DEM for site: {site_name}")
                            dem_path = self.download_dem_for_aoi(
                                aoi_name=site_name,
                                bounds=bounds,
                                dem_type='SRTMGL1',
                                buffer_degrees=0.02
                            )
                            results[site_name] = dem_path
                
            except Exception as e:
                logger.error(f"Error processing bounds file {bounds_file}: {e}")
        
        logger.info(f"Downloaded DEMs for {len(results)} sites")
        return results

# Example usage and testing
if __name__ == "__main__":
    # Test the downloader
    api_key = os.environ.get('OPENTOPOGRAPHY_API_KEY')
    if not api_key:
        print("Error: OPENTOPOGRAPHY_API_KEY environment variable not set")
        sys.exit(1)
    
    downloader = AOIDEMDownloader(api_key)
    
    # Test with a small area
    test_bounds = (18.0, 4.0, 18.1, 4.1)  # Small area in Central Africa
    dem_path = downloader.download_dem_for_aoi(
        aoi_name="test_area",
        bounds=test_bounds,
        dem_type='SRTMGL1'
    )
    
    if dem_path:
        print(f"✅ Test successful: {dem_path}")
    else:
        print("❌ Test failed")
        sys.exit(1)