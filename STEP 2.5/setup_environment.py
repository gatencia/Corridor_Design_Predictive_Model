#!/usr/bin/env python3
"""
Step 2.5 Setup and Environment Validation
Prepares the environment and validates all dependencies for DEM acquisition.
"""

import sys
import subprocess
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version >= (3, 8):
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_required_packages():
    """Check required Python packages."""
    print("📦 Checking required packages...")
    
    required_packages = [
        ('requests', 'HTTP requests for downloading'),
        ('geopandas', 'Geospatial data processing'),
        ('pathlib', 'Path handling'),
        ('concurrent.futures', 'Concurrent downloads'),
        ('json', 'JSON processing'),
        ('logging', 'Logging system')
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install requests geopandas")
        return False
    
    return True

def check_internet_connectivity():
    """Check internet connectivity to SRTM sources."""
    print("🌐 Checking internet connectivity...")
    
    test_urls = [
        ("OpenTopography S3", "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1"),
        ("USGS SRTM", "https://dds.cr.usgs.gov/srtm/version2_1/SRTM3"),
        ("Google DNS", "https://8.8.8.8")  # Basic connectivity test
    ]
    
    import requests
    
    connectivity_ok = False
    
    for name, url in test_urls:
        try:
            response = requests.head(url, timeout=10)
            if response.status_code < 400:
                print(f"   ✅ {name} - accessible")
                connectivity_ok = True
            else:
                print(f"   ⚠️  {name} - HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"   ❌ {name} - {e}")
    
    if connectivity_ok:
        print("   ✅ Internet connectivity confirmed")
        return True
    else:
        print("   ❌ No SRTM sources accessible")
        return False

def setup_directory_structure():
    """Create required directory structure."""
    print("📁 Setting up directory structure...")
    
    directories = [
        "outputs/aoi_specific_dems",
        "logs",
        "inputs/aois",
        "temp"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    return True

def check_step2_outputs():
    """Check for Step 2 AOI outputs."""
    print("🔍 Checking for Step 2 outputs...")
    
    # Possible Step 2 output locations
    possible_paths = [
        Path("../STEP 2/data/outputs"),
        Path("../STEP 2/outputs"),
        Path("inputs/aois"),
        Path("../../STEP 2/data/outputs")
    ]
    
    aoi_files_found = []
    
    for path in possible_paths:
        if path.exists():
            print(f"   📂 Checking: {path}")
            
            # Look for AOI files
            patterns = ["*aoi*.geojson", "*aoi*.shp", "*AOI*.geojson", "*AOI*.shp"]
            for pattern in patterns:
                files = list(path.rglob(pattern))
                aoi_files_found.extend(files)
            
            if aoi_files_found:
                print(f"   ✅ Found {len(aoi_files_found)} AOI files in {path}")
                for aoi_file in aoi_files_found[:3]:  # Show first 3
                    print(f"     - {aoi_file.name}")
                if len(aoi_files_found) > 3:
                    print(f"     ... and {len(aoi_files_found) - 3} more")
                return True
    
    print("   ❌ No Step 2 AOI outputs found")
    print("   Ensure Step 2 has been completed successfully")
    return False

def test_srtm_download():
    """Test downloading a single SRTM tile."""
    print("🏔️  Testing SRTM download capability...")
    
    import requests
    
    # Test with a small, reliable tile from AWS Open Data (Skadi)
    # N08E013.hgt for Cameroon
    lat_band = "N08"
    filename = "N08E013.hgt"
    test_url = f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/{lat_band}/{filename}"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"   🔄 Testing download from: {test_url}")
            
            # Try to download first 1KB as test
            headers = {'Range': 'bytes=0-1023'}
            response = requests.get(test_url, headers=headers, timeout=20) # Increased timeout
            
            if response.status_code in [200, 206]:  # OK or Partial Content
                print(f"   ✅ SRTM download test successful ({len(response.content)} bytes)")
                return True
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.reason}")
                # Attempt to get more error details if available
                try:
                    error_content = response.json() # Or response.text
                    print(f"      Error content: {error_content}")
                except Exception:
                    pass # Ignore if no JSON/text content
                return False
                
    except requests.RequestException as e:
        print(f"   ❌ Download test failed: {e}")
        return False

def create_sample_aoi():
    """Create a sample AOI file for testing if none exist."""
    print("📍 Creating sample AOI for testing...")
    
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        # Create sample AOI covering central Cameroon
        sample_polygon = Polygon([
            (13.0, 8.0),   # SW corner
            (13.5, 8.0),   # SE corner  
            (13.5, 8.5),   # NE corner
            (13.0, 8.5),   # NW corner
            (13.0, 8.0)    # Close polygon
        ])
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame([{
            'study_site': 'Sample_Test_Site',
            'area_km2': 2500.0,
            'buffer_km': 5.0,
            'created': datetime.now().isoformat()
        }], geometry=[sample_polygon], crs='EPSG:4326')
        
        # Save to inputs directory
        sample_file = Path("inputs/aois/sample_test_aoi.geojson")
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        
        gdf.to_file(sample_file, driver='GeoJSON')
        
        print(f"   ✅ Created sample AOI: {sample_file}")
        print(f"   🗺️  Coverage: Central Cameroon (13.0-13.5°E, 8.0-8.5°N)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create sample AOI: {e}")
        return False

def run_comprehensive_validation():
    """Run complete environment validation."""
    print("🔧 STEP 2.5 Environment Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Directory Structure", setup_directory_structure),
        ("Internet Connectivity", check_internet_connectivity),
        ("SRTM Download Test", test_srtm_download),
        ("Step 2 Outputs", check_step2_outputs)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Check failed: {e}")
            results.append(False)
    
    # If no Step 2 outputs found, offer to create sample
    if not results[-1]:  # Step 2 outputs check failed
        print(f"\n📍 No Step 2 outputs found - creating sample AOI...")
        sample_created = create_sample_aoi()
        if sample_created:
            results[-1] = True  # Update Step 2 check result
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Validation Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Environment validation complete - ready for DEM acquisition!")
        
        print(f"\n🚀 Next Steps:")
        print(f"1. Run DEM acquisition: python step25_dem_downloader.py")
        print(f"2. Or run tests first: python step25_config.py --test")
        
        return True
    else:
        print("❌ Environment validation failed - address issues above")
        
        print(f"\n🔧 Common fixes:")
        print(f"• Install packages: pip install requests geopandas")
        print(f"• Check internet connection")  
        print(f"• Complete Step 2 GPS processing first")
        
        return False

def main():
    """Main setup function."""
    success = run_comprehensive_validation()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)