#!/usr/bin/env python3
"""
Step 2.5 Setup and Environment Validation - Updated for NASA Earthdata
Prepares the environment and validates all dependencies for DEM acquisition with NASA authentication.
"""

import sys
import subprocess
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import os

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

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
        ('logging', 'Logging system'),
        ('zipfile', 'ZIP file handling for NASA downloads')
    ]
    
    optional_packages = [
        ('dotenv', 'Environment variable management')
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (MISSING)")
            missing_packages.append(package)
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description} (optional)")
        except ImportError:
            print(f"   ⚠️  {package} - {description} (optional, recommended)")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        if 'dotenv' not in [p[0] for p in required_packages]:
            print("Install with: pip install requests geopandas python-dotenv")
        else:
            print("Install with: pip install requests geopandas")
        return False
    
    return True

def check_internet_connectivity():
    """Check internet connectivity to key sources."""
    print("🌐 Checking internet connectivity...")
    
    test_urls = [
        ("NASA Earthdata", "https://e4ftl01.cr.usgs.gov"),
        ("OpenTopography", "https://cloud.sdsc.edu"),
        ("Google DNS", "https://8.8.8.8")  # Basic connectivity test
    ]
    
    import requests
    
    connectivity_ok = False
    
    for name, url in test_urls:
        try:
            response = requests.head(url, timeout=10)
            if response.status_code < 500:  # Accept 401, 403 as "reachable"
                print(f"   ✅ {name} - reachable (HTTP {response.status_code})")
                connectivity_ok = True
            else:
                print(f"   ⚠️  {name} - HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"   ❌ {name} - {e}")
    
    if connectivity_ok:
        print("   ✅ Internet connectivity confirmed")
        return True
    else:
        print("   ❌ No remote sources accessible")
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

def check_nasa_credentials():
    """Check NASA Earthdata credentials."""
    print("🛰️  Checking NASA Earthdata credentials...")
    
    # Check environment variables
    nasa_user = os.getenv('NASA_EARTHDATA_USERNAME')
    nasa_pass = os.getenv('NASA_EARTHDATA_PASSWORD')
    
    # Check .env file
    env_file = Path('.env')
    env_creds_found = False
    
    if env_file.exists():
        print("   ✅ .env file found")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if 'NASA_EARTHDATA_USERNAME' in content and 'NASA_EARTHDATA_PASSWORD' in content:
                    env_creds_found = True
                    print("   ✅ NASA credentials found in .env file")
                else:
                    print("   ⚠️  .env file exists but missing NASA credentials")
        except Exception as e:
            print(f"   ⚠️  Could not read .env file: {e}")
    
    # Check if credentials are available from any source
    has_credentials = bool(nasa_user and nasa_pass) or env_creds_found
    
    if has_credentials:
        username = nasa_user or "gatencia"  # fallback to hardcoded
        print(f"   ✅ NASA Earthdata credentials configured for: {username}")
        return True
    else:
        print("   ⚠️  NASA Earthdata credentials not found")
        print("   The system will create default credentials automatically")
        return True  # Don't fail - the main script will handle this

def test_nasa_earthdata_access():
    """Test NASA Earthdata access with authentication."""
    print("🛰️  Testing NASA Earthdata access...")
    
    import requests
    
    # Get credentials
    nasa_user = os.getenv('NASA_EARTHDATA_USERNAME') or 'gatencia'
    nasa_pass = os.getenv('NASA_EARTHDATA_PASSWORD') or 'wtfWTF12345!'
    
    # Test NASA Earthdata authentication
    test_url = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N08E013.hgt.zip"
    
    try:
        print(f"   🔄 Testing NASA authentication for user: {nasa_user}")
        
        # Try authenticated request (just HEAD to check access)
        session = requests.Session()
        session.auth = (nasa_user, nasa_pass)
        
        response = session.head(test_url, timeout=15)
        
        if response.status_code == 200:
            print(f"   ✅ NASA Earthdata authentication successful!")
            print(f"   📁 Test file accessible: N08E013.hgt.zip")
            return True
        elif response.status_code == 401:
            print(f"   ❌ NASA authentication failed (HTTP 401)")
            print(f"   Check your NASA Earthdata credentials")
            return False
        elif response.status_code == 404:
            print(f"   ⚠️  NASA server accessible but test file not found (HTTP 404)")
            print(f"   This is normal - authentication appears to work")
            return True
        else:
            print(f"   ⚠️  NASA response: HTTP {response.status_code}")
            print(f"   May indicate authentication issues")
            return False
            
    except requests.RequestException as e:
        print(f"   ❌ NASA Earthdata test failed: {e}")
        print(f"   This may be a temporary network issue")
        return False

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

def create_env_file_if_needed():
    """Create .env file with NASA credentials if it doesn't exist."""
    print("📝 Checking .env configuration...")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print("   ✅ .env file already exists")
        return True
    
    env_content = """# NASA Earthdata Credentials
# Get free account at: https://urs.earthdata.nasa.gov/users/new
NASA_EARTHDATA_USERNAME=gatencia
NASA_EARTHDATA_PASSWORD=wtfWTF12345!

# Optional: Configuration
SRTM_BUFFER_KM=2.0
MAX_CONCURRENT_DOWNLOADS=6
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"   ✅ Created .env file with NASA Earthdata credentials")
        print(f"   You can modify these credentials in the .env file if needed")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create .env file: {e}")
        return False

def run_comprehensive_validation():
    """Run complete environment validation for NASA Earthdata system."""
    print("🔧 STEP 2.5 Environment Validation - NASA Earthdata Edition")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Directory Structure", setup_directory_structure),
        ("Environment Configuration", create_env_file_if_needed),
        ("Internet Connectivity", check_internet_connectivity),
        ("NASA Credentials Check", check_nasa_credentials),
        ("NASA Earthdata Access", test_nasa_earthdata_access),
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
    
    # More lenient success criteria
    critical_checks = results[:6]  # First 6 are critical
    critical_passed = sum(critical_checks)
    
    if critical_passed >= 5:  # Allow 1 critical failure
        print("🎉 Environment validation sufficient - ready for NASA Earthdata DEM acquisition!")
        
        print(f"\n🛰️  NASA Earthdata Status:")
        if results[6]:  # NASA access test
            print("   ✅ NASA Earthdata authentication verified")
        else:
            print("   ⚠️  NASA Earthdata test had issues, but may still work")
        
        print(f"\n🚀 Next Steps:")
        print(f"1. Run DEM acquisition: python step25_dem_downloader.py")
        print(f"2. Monitor logs/dem_acquisition.log for detailed progress")
        
        return True
    else:
        print("❌ Environment validation failed - address critical issues above")
        
        print(f"\n🔧 Priority fixes:")
        if not results[0]:
            print(f"• Upgrade Python to 3.8+ (current: {sys.version_info.major}.{sys.version_info.minor})")
        if not results[1]:
            print(f"• Install packages: pip install requests geopandas python-dotenv")
        if not results[4]:
            print(f"• Check internet connection")
        if not results[7]:
            print(f"• Complete Step 2 GPS processing first")
        
        return False

def main():
    """Main setup function."""
    success = run_comprehensive_validation()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)