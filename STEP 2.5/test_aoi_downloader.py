#!/usr/bin/env python3
"""
Test the fixed AOI DEM downloader with the correct OpenTopography API configuration.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the fixed downloader
from aoi_dem_downloader import AOIDEMDownloader

def test_fixed_api_configuration():
    """Test the fixed API configuration with small test areas."""
    
    print("🔧 Testing Fixed OpenTopography API Configuration")
    print("=" * 60)
    
    # Get API key
    api_key = os.environ.get('OPENTOPOGRAPHY_API_KEY')
    if not api_key:
        print("❌ Error: OPENTOPOGRAPHY_API_KEY environment variable not set")
        return False
    
    print(f"✅ API key loaded: {api_key[:8]}...")
    
    # Initialize downloader
    downloader = AOIDEMDownloader(api_key)
    
    # Test areas (small for quick testing)
    test_areas = {
        'Central_Africa_Small': (18.0, 4.0, 18.2, 4.2),  # Small test area in Central Africa
        'Cameroon_Test': (10.0, 5.0, 10.1, 5.1)  # Small area in Cameroon
    }
    
    results = {}
    
    for area_name, bounds in test_areas.items():
        print(f"\n🌍 Testing area: {area_name}")
        print(f"   Bounds: {bounds}")
        
        # Test both DEM types
        for dem_type in ['SRTMGL1', 'SRTMGL3']:
            print(f"\n📡 Downloading {dem_type} for {area_name}...")
            
            try:
                dem_path = downloader.download_dem_for_aoi(
                    aoi_name=f"{area_name}_test",
                    bounds=bounds,
                    dem_type=dem_type,
                    buffer_degrees=0.01
                )
                
                if dem_path and dem_path.exists():
                    file_size = dem_path.stat().st_size
                    print(f"   ✅ SUCCESS: {dem_path.name}")
                    print(f"   📁 Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                    
                    results[f"{area_name}_{dem_type}"] = {
                        'success': True,
                        'path': str(dem_path),
                        'size_bytes': file_size
                    }
                else:
                    print(f"   ❌ FAILED: No file downloaded")
                    results[f"{area_name}_{dem_type}"] = {
                        'success': False,
                        'error': 'No file downloaded'
                    }
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                results[f"{area_name}_{dem_type}"] = {
                    'success': False,
                    'error': str(e)
                }
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r['success'])
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    print(f"\n✅ SUCCESSFUL DOWNLOADS:")
    for test_name, result in results.items():
        if result['success']:
            size_mb = result['size_bytes'] / 1024 / 1024
            print(f"   ✓ {test_name}: {size_mb:.2f} MB")
    
    print(f"\n❌ FAILED DOWNLOADS:")
    for test_name, result in results.items():
        if not result['success']:
            print(f"   ✗ {test_name}: {result.get('error', 'Unknown error')}")
    
    # Validate at least one success
    if successful_tests > 0:
        print(f"\n🎉 API configuration is working! At least {successful_tests} download(s) succeeded.")
        return True
    else:
        print(f"\n❌ API configuration still has issues. No downloads succeeded.")
        return False

if __name__ == "__main__":
    success = test_fixed_api_configuration()
    sys.exit(0 if success else 1)