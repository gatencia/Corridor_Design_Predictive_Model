"""
Fixed test script for DEM acquisition system.
Tests tile calculation and download functionality.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

def test_tile_calculation():
    """Test SRTM tile calculation logic."""
    print("🧪 Testing tile calculation...")
    
    # Import the main module (assuming it's saved as step25_dem_downloader.py)
    try:
        from step25_dem_downloader import SRTMTileCalculator, AOIBounds, SRTMTile
    except ImportError:
        print("❌ Could not import DEM downloader module")
        return False
    
    # Create test AOI (small area in Cameroon)
    test_aoi = AOIBounds(
        name="Test_Cameroon",
        min_lat=8.0,
        max_lat=8.5,
        min_lon=13.0,
        max_lon=13.5,
        source_file="test.geojson"
    )
    
    # Calculate tiles
    calculator = SRTMTileCalculator(buffer_km=1.0)
    tiles = calculator.calculate_tiles_for_aoi(test_aoi)
    
    print(f"   Test AOI requires {len(tiles)} tiles:")
    for tile in sorted(tiles, key=lambda t: (t.lat, t.lon)):
        print(f"     {tile.filename} -> {tile.nasa_url}")  # FIX: Changed from s3_url to nasa_url
    
    # Verify tile naming
    expected_tiles = {"N08E013.hgt", "N08E014.hgt", "N09E013.hgt", "N09E014.hgt"}  # With 1km buffer
    actual_filenames = {tile.filename for tile in tiles}
    
    if expected_tiles.issubset(actual_filenames):
        print("   ✅ Tile calculation correct")
        return True
    else:
        print(f"   ❌ Expected {expected_tiles}, got {actual_filenames}")
        return False

def test_single_tile_download():
    """Test downloading a single SRTM tile."""
    print("🧪 Testing single tile download...")
    
    try:
        from step25_dem_downloader import SRTMDownloader, SRTMTile
    except ImportError:
        print("❌ Could not import DEM downloader module")
        return False
    
    # Create temporary download directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with a small, reliable tile (middle of Cameroon)
        test_tile = SRTMTile(lat=8, lon=13)  # N08E013.hgt
        
        downloader = SRTMDownloader(
            output_dir=temp_path,
            max_workers=1,
            max_retries=2,
            timeout_seconds=15
        )
        
        print(f"   Attempting to download: {test_tile.filename}")
        print(f"   From: {test_tile.nasa_url}")  # FIX: Changed from s3_url to nasa_url
        
        success = downloader._download_tile_with_retry(test_tile)
        
        if success:
            downloaded_file = temp_path / test_tile.filename
            if downloaded_file.exists():
                size_mb = downloaded_file.stat().st_size / (1024 * 1024)
                print(f"   ✅ Download successful: {size_mb:.2f} MB")
                return True
            else:
                print("   ❌ File not found after download")
                return False
        else:
            print("   ❌ Download failed")
            return False

def test_aoi_discovery():
    """Test Step 2 AOI file discovery."""
    print("🧪 Testing AOI discovery...")
    
    try:
        from step25_dem_downloader import AOIProcessor
    except ImportError:
        print("❌ Could not import DEM downloader module")
        return False
    
    # Try to discover actual Step 2 outputs
    try:
        processor = AOIProcessor()
        aoi_files = processor.discover_aoi_files()
        
        if aoi_files:
            print(f"   ✅ Found {len(aoi_files)} AOI files:")
            for aoi_file in aoi_files[:3]:  # Show first 3
                print(f"     - {aoi_file.name}")
            if len(aoi_files) > 3:
                print(f"     ... and {len(aoi_files) - 3} more")
            
            # Try to load bounds from first file
            try:
                aoi_bounds = processor.load_aoi_bounds([aoi_files[0]])
                if aoi_bounds:
                    bounds = aoi_bounds[0]
                    print(f"   ✅ Sample AOI: {bounds.name}")
                    print(f"     Bounds: {bounds.min_lat:.3f},{bounds.min_lon:.3f} to {bounds.max_lat:.3f},{bounds.max_lon:.3f}")
                    return True
                else:
                    print("   ❌ Could not load AOI bounds")
                    return False
            except Exception as e:
                print(f"   ❌ Error loading AOI bounds: {e}")
                return False
        else:
            print("   ⚠️  No AOI files found - ensure Step 2 has been completed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during AOI discovery: {e}")
        return False

def test_dataclass_hashability():
    """Test that SRTMTile is properly hashable."""
    print("🧪 Testing dataclass hashability...")
    
    try:
        from step25_dem_downloader import SRTMTile
        
        # Create test tiles
        tile1 = SRTMTile(lat=8, lon=13)
        tile2 = SRTMTile(lat=8, lon=14)
        tile3 = SRTMTile(lat=8, lon=13)  # Same as tile1
        
        # Test that they can be added to a set
        tile_set = {tile1, tile2, tile3}
        
        if len(tile_set) == 2:  # tile1 and tile3 should be the same
            print("   ✅ SRTMTile is properly hashable")
            print(f"     Set contains: {[t.filename for t in tile_set]}")
            return True
        else:
            print(f"   ❌ Expected 2 unique tiles, got {len(tile_set)}")
            return False
            
    except TypeError as e:
        print(f"   ❌ SRTMTile is not hashable: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error testing hashability: {e}")
        return False

def run_quick_test():
    """Run all quick tests."""
    print("🚀 Step 2.5 DEM Acquisition - Fixed Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dataclass Hashability", test_dataclass_hashability),
        ("Tile Calculation", test_tile_calculation),
        ("AOI Discovery", test_aoi_discovery),
        ("Single Tile Download", test_single_tile_download)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Fixed DEM acquisition system is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check issues above.")
        return False

# Standalone execution scripts
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 2.5 DEM Acquisition Tools")
    parser.add_argument("--test", action="store_true", help="Run quick test suite")
    parser.add_argument("--config", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.test:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    elif args.config:
        # Try to import config if available
        try:
            from config import DEMConfig
            print("Step 2.5 DEM Acquisition Configuration:")
            print("=" * 40)
            for attr_name in dir(DEMConfig):
                if not attr_name.startswith('_'):
                    value = getattr(DEMConfig, attr_name)
                    print(f"{attr_name}: {value}")
        except ImportError:
            print("Configuration module not available")
    
    else:
        print("Step 2.5 DEM Acquisition System - Fixed Version")
        print("Usage:")
        print("  python test_system.py --test     # Run test suite")
        print("  python test_system.py --config   # Show configuration")
        print("\nTo run full DEM acquisition:")
        print("  python step25_dem_downloader.py")