#!/usr/bin/env python3
"""
Simple test for Step 2 GPS processing modules.
Run this from the STEP 2 directory.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path.cwd()
if current_dir.name == 'tests':
    # If running from tests directory, go up one level
    current_dir = current_dir.parent

src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_step2_imports():
    """Test Step 2 specific imports."""
    
    print("🧪 Testing Step 2 GPS Processing Imports")
    print("=" * 50)
    print(f"📂 Working directory: {current_dir}")
    print(f"📁 Source directory: {src_dir}")
    
    # Test required libraries first
    print("\n1. Testing required libraries...")
    required_libs = [
        'pandas',
        'geopandas', 
        'numpy',
        'shapely',
        'pyproj',
        'geopy',
        'tqdm'
    ]
    
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"   ✅ {lib}")
        except ImportError:
            print(f"   ❌ {lib} (missing)")
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"\n⚠️  Missing libraries: {', '.join(missing_libs)}")
        print("Install with: pip install pandas geopandas numpy shapely pyproj geopy tqdm")
        return False
    
    # Test main GPS processor
    print("\n2. Testing GPS data processor...")
    try:
        from data_ingestion import GPSDataProcessor
        processor = GPSDataProcessor()
        print("   ✅ GPSDataProcessor imported and created successfully")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test data directory
    print("\n3. Checking for GPS data...")
    data_paths = [
        current_dir / ".." / "GPS_Collar_CSV_Mark",
        Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
    ]
    
    data_dir = None
    for path in data_paths:
        if path.exists():
            data_dir = path
            break
    
    if data_dir:
        csv_files = list(data_dir.glob("*.csv"))
        print(f"   ✅ Found {len(csv_files)} CSV files in {data_dir}")
        if csv_files:
            print(f"   📄 Sample files:")
            for file in sorted(csv_files)[:3]:
                print(f"      • {file.name}")
    else:
        print("   ⚠️  GPS data directory not found")
        print("   You'll need to update the path in run_gps_processing.py")
    
    print(f"\n🎉 Step 2 imports working correctly!")
    return True

def run_quick_test():
    """Run a quick processing test if data is available."""
    
    print(f"\n🚀 Running Quick Processing Test")
    print("=" * 40)
    
    try:
        from data_ingestion import GPSDataProcessor
        
        # Find a single test file
        data_paths = [
            current_dir / ".." / "GPS_Collar_CSV_Mark",
            Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
        ]
        
        test_file = None
        for data_dir in data_paths:
            if data_dir.exists():
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    test_file = csv_files[0]  # Use first file for test
                    break
        
        if test_file:
            print(f"📄 Testing with: {test_file.name}")
            
            processor = GPSDataProcessor()
            
            # Try to load just one file
            gdf = processor.load_gps_data(test_file)
            
            print(f"   ✅ Successfully loaded {len(gdf)} GPS fixes")
            print(f"   📊 Individuals: {gdf['individual-local-identifier'].nunique()}")
            print(f"   📅 Date range: {gdf['timestamp'].min().date()} to {gdf['timestamp'].max().date()}")
            
            # Test AOI generation
            aoi_gdf = processor.generate_aoi(gdf, buffer_km=2.0)
            print(f"   🗺️  Generated AOI: {aoi_gdf['area_km2'].iloc[0]:.1f} km²")
            
            print(f"   🎉 Quick test successful!")
            return True
            
        else:
            print("   ⚠️  No test data available - skipping processing test")
            return True
            
    except Exception as e:
        print(f"   ❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("🐘 Step 2 GPS Processing - Simple Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_step2_imports()
    
    if imports_ok:
        # Run quick processing test
        processing_ok = run_quick_test()
        
        if processing_ok:
            print(f"\n✅ All tests passed! Ready to run full GPS processing.")
            print(f"Next step: python run_gps_processing.py")
        else:
            print(f"\n⚠️  Imports work but processing test failed.")
            print(f"Check the GPS data files and try running the full script.")
    else:
        print(f"\n❌ Import tests failed. Fix the issues above first.")