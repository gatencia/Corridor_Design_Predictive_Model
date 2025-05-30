#!/usr/bin/env python3
"""
Quick DEM Download Runner for Step 2.5
Gets you DEM data FAST with the smallest possible AOI
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def main():
    """Run the complete DEM download process with minimum viable AOI."""
    
    print("🚀 STEP 2.5 - Quick DEM Download")
    print("=" * 40)
    print("Goal: Get DEM data for the SMALLEST region to prove the pipeline works")
    print()
    
    # 1. Load environment variables from .env if it exists
    env_file = Path(".env")
    if env_file.exists():
        print("📁 Loading credentials from .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Loaded environment variables")
    else:
        print("⚠️  No .env file found - will try with existing environment variables")
    
    # 2. Check if we have at least one credential
    has_credentials = any([
        os.environ.get('OPENTOPOGRAPHY_API_KEY'),
        os.environ.get('EARTHDATA_USER'),
        os.environ.get('COPERNICUS_TOKEN')
    ])
    
    if not has_credentials:
        print("\n❌ No DEM service credentials found!")
        print("Run this first: python setup_credentials.py")
        print("Or set environment variables manually:")
        print("  export OPENTOPOGRAPHY_API_KEY='your_key_here'")
        return False
    
    # 3. Import and run the robust downloader
    print("\n📡 Starting robust DEM downloader...")
    
    try:
        from robust_dem_downloader import RobustDEMDownloader
        
        # Initialize downloader
        downloader = RobustDEMDownloader()
        
        # Create the smallest possible test AOI
        test_name, test_bounds = downloader.create_test_aoi()
        
        print(f"\n🎯 Target: {test_name}")
        print(f"Bounds: {test_bounds}")
        print(f"Size: ~11km x 11km (perfect for quick testing)")
        
        # Download with all fallback sources
        dem_path = downloader.download_dem_robust(
            name=test_name,
            bounds=test_bounds,
            target_resolution=30
        )
        
        if dem_path:
            print(f"\n🎉 SUCCESS! Downloaded DEM: {dem_path}")
            print(f"File size: {dem_path.stat().st_size:,} bytes")
            
            # Quick validation
            try:
                import rasterio
                with rasterio.open(dem_path) as src:
                    print(f"Dimensions: {src.width} x {src.height} pixels")
                    print(f"Resolution: ~{abs(src.transform[0]):.4f}° per pixel")
                    print(f"CRS: {src.crs}")
                    
                    # Read elevation stats
                    data = src.read(1)
                    valid_data = data[data != src.nodata]
                    if len(valid_data) > 0:
                        print(f"Elevation range: {valid_data.min():.1f} to {valid_data.max():.1f} meters")
                        print(f"Mean elevation: {valid_data.mean():.1f} meters")
                    
            except Exception as e:
                print(f"Could not read detailed stats: {e}")
            
            print(f"\n✅ STEP 2.5 COMPLETE!")
            print(f"DEM file ready for Step 3 EnergyScape processing")
            print(f"Location: {dem_path}")
            
            return True
            
        else:
            print("\n💥 FAILED: Could not download DEM from any source")
            print("\nTroubleshooting:")
            print("1. Check your internet connection")
            print("2. Verify credentials: python setup_credentials.py")
            print("3. Try a different region (some areas have data gaps)")
            
            return False
            
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install rasterio requests numpy")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Check the logs above for more details")
        return False

if __name__ == "__main__":
    print("🌍 Quick DEM Download for Corridor Design")
    print("Getting elevation data for the smallest possible test region...")
    print()
    
    success = main()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. ✅ Step 2.5 DEM download - COMPLETE")
        print("2. 🔄 Ready for Step 3 EnergyScape processing")
        print("3. 📊 Run: cd ../STEP\\ 3 && python run_energyscape.py")
    else:
        print("\n🔧 Fix the issues above and try again")
        sys.exit(1)