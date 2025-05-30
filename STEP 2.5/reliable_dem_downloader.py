#!/usr/bin/env python3
"""
Reliable DEM Downloader for Elephant Corridor Research
Uses the elevation library for robust SRTM data download without API keys
"""

import os
import sys
import json
import logging
from pathlib import Path
import elevation
import pandas as pd
from typing import List, Tuple, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reliable_dem_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure all necessary directories exist"""
    dirs_to_create = [
        'outputs/reliable_dems',
        'logs',
        'data/temp'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory ensured: {dir_path}")

def load_aoi_summary() -> List[Dict[str, Any]]:
    """Load AOI summary from Step 2"""
    summary_path = "../STEP 2/outputs/aoi_summary.json"
    
    if not os.path.exists(summary_path):
        logger.error(f"❌ AOI summary not found at: {summary_path}")
        logger.error("Please run Step 2 first to generate the AOI summary")
        sys.exit(1)
    
    with open(summary_path, 'r') as f:
        aoi_data = json.load(f)
    
    logger.info(f"✓ Loaded {len(aoi_data)} AOIs from Step 2")
    return aoi_data

def safe_filename(name: str) -> str:
    """Convert AOI name to safe filename"""
    # Replace spaces and special characters
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    # Remove multiple underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    return safe_name.strip('_')

def download_dem_for_aoi(aoi: Dict[str, Any]) -> bool:
    """
    Download DEM for a single AOI using elevation library
    Returns True if successful, False otherwise
    """
    name = aoi['name']
    bounds = aoi['bounds']
    
    logger.info(f"DOWNLOADING: {name}")
    
    # Add 0.02 degree buffer (roughly 2km)
    buffer = 0.018  # Slightly smaller buffer to avoid hitting tile limits
    min_lon = bounds['west'] - buffer
    max_lon = bounds['east'] + buffer
    min_lat = bounds['south'] - buffer
    max_lat = bounds['north'] + buffer
    
    logger.info(f"  Original bounds: {bounds['south']:.4f},{bounds['west']:.4f} to {bounds['north']:.4f},{bounds['east']:.4f}")
    logger.info(f"  Buffered bounds: {min_lat:.4f},{min_lon:.4f} to {max_lat:.4f},{max_lon:.4f}")
    
    # Check if area is too large (elevation library has limits)
    area_degrees = (max_lon - min_lon) * (max_lat - min_lat)
    if area_degrees > 20:  # Arbitrary limit to avoid "too many tiles" error
        logger.error(f"  Error: Area too large ({area_degrees:.1f} sq degrees). Skipping {name}")
        return False
    
    # Create safe filename
    safe_name = safe_filename(name)
    output_path = f"outputs/reliable_dems/dem_{safe_name}.tif"
    
    try:
        # Use elevation library to download and clip DEM
        elevation.clip(
            bounds=(min_lon, min_lat, max_lon, max_lat),
            output=output_path,
            product='SRTM1'  # 30m resolution
        )
        
        # Verify the file was created and has reasonable size
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 1000:  # At least 1KB
                logger.info(f"  ✅ Success: {output_path} ({file_size:,} bytes)")
                return True
            else:
                logger.error(f"  ❌ File too small: {output_path} ({file_size} bytes)")
                return False
        else:
            logger.error(f"  ❌ Output file not created: {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"  Error downloading DEM for {name}: {str(e)}")
        return False

def clean_elevation_cache():
    """Clean up elevation library cache if needed"""
    try:
        # The elevation library uses a cache, we can clean it if needed
        elevation.clean()
        logger.info("✓ Cleaned elevation cache")
    except Exception as e:
        logger.warning(f"Could not clean elevation cache: {e}")

def main():
    """Main execution function"""
    logger.info("🗺️  Starting Reliable DEM Download Process")
    logger.info("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Load AOI data from Step 2
    aoi_data = load_aoi_summary()
    
    # Track results
    successful_downloads = []
    failed_downloads = []
    
    # Process each AOI
    for i, aoi in enumerate(aoi_data, 1):
        logger.info(f"\n📄 Processing {i}/{len(aoi_data)}: {aoi['name']}")
        
        success = download_dem_for_aoi(aoi)
        
        if success:
            successful_downloads.append(aoi['name'])
            logger.info(f"✅ Success: {aoi['name']}")
        else:
            failed_downloads.append(aoi['name'])
            logger.error(f"❌ Failed: {aoi['name']}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successful: {len(successful_downloads)}/{len(aoi_data)}")
    logger.info(f"❌ Failed: {len(failed_downloads)}/{len(aoi_data)}")
    
    if successful_downloads:
        logger.info(f"\n✅ Successful downloads:")
        for name in successful_downloads:
            logger.info(f"  - {name}")
    
    if failed_downloads:
        logger.info(f"\n❌ Failed downloads:")
        for name in failed_downloads:
            logger.info(f"  - {name}")
    
    # Save results
    results = {
        'successful_downloads': successful_downloads,
        'failed_downloads': failed_downloads,
        'total_aois': len(aoi_data),
        'success_rate': len(successful_downloads) / len(aoi_data) if aoi_data else 0
    }
    
    with open('outputs/reliable_dem_download_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: outputs/reliable_dem_download_results.json")
    logger.info("🏁 DEM download process complete!")
    
    return len(successful_downloads) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)