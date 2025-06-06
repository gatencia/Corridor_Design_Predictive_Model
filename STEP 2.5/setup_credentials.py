#!/usr/bin/env python3
"""
Credential Setup for Multi-Source DEM Downloader
Helps set up credentials for all DEM sources
"""

import os
from pathlib import Path
import getpass

def setup_credentials():
    """Interactive setup of credentials for all DEM sources."""
    
    print("🔐 Setting up credentials for DEM sources...")
    print("=" * 50)
    
    env_file = Path(".env")
    credentials = {}
    
    # OpenTopography API Key
    print("\n1. OpenTopography API Key")
    print("   • Go to: https://portal.opentopography.org/")
    print("   • Sign in → Profile → 'Request API Key'")
    current_ot_key = os.environ.get('OPENTOPOGRAPHY_API_KEY')
    if current_ot_key:
        print(f"   • Current key: {current_ot_key[:10]}...")
        use_current = input("   Use current key? (y/n): ").lower().startswith('y')
        if use_current:
            credentials['OPENTOPOGRAPHY_API_KEY'] = current_ot_key
        else:
            new_key = getpass.getpass("   Enter new OpenTopography API key: ")
            if new_key.strip():
                credentials['OPENTOPOGRAPHY_API_KEY'] = new_key.strip()
    else:
        ot_key = getpass.getpass("   Enter OpenTopography API key (or skip): ")
        if ot_key.strip():
            credentials['OPENTOPOGRAPHY_API_KEY'] = ot_key.strip()
    
    # NASA Earthdata
    print("\n2. NASA Earthdata Credentials")
    print("   • Go to: https://urs.earthdata.nasa.gov/users/new")
    print("   • Create free account")
    
    current_nasa_user = os.environ.get('EARTHDATA_USER')
    if current_nasa_user:
        print(f"   • Current user: {current_nasa_user}")
        use_current = input("   Use current credentials? (y/n): ").lower().startswith('y')
        if use_current:
            credentials['EARTHDATA_USER'] = current_nasa_user
            credentials['EARTHDATA_PASS'] = os.environ.get('EARTHDATA_PASS', '')
        else:
            nasa_user = input("   Enter NASA Earthdata username: ")
            nasa_pass = getpass.getpass("   Enter NASA Earthdata password: ")
            if nasa_user.strip() and nasa_pass.strip():
                credentials['EARTHDATA_USER'] = nasa_user.strip()
                credentials['EARTHDATA_PASS'] = nasa_pass.strip()
    else:
        nasa_user = input("   Enter NASA Earthdata username (or skip): ")
        if nasa_user.strip():
            nasa_pass = getpass.getpass("   Enter NASA Earthdata password: ")
            if nasa_pass.strip():
                credentials['EARTHDATA_USER'] = nasa_user.strip()
                credentials['EARTHDATA_PASS'] = nasa_pass.strip()
    
    # Copernicus Data Space
    print("\n3. Copernicus Data Space Token")
    print("   • Go to: https://dataspace.copernicus.eu/")
    print("   • Register → Generate OAuth access token")
    
    current_cop_token = os.environ.get('COPERNICUS_TOKEN')
    if current_cop_token:
        print(f"   • Current token: {current_cop_token[:20]}...")
        use_current = input("   Use current token? (y/n): ").lower().startswith('y')
        if use_current:
            credentials['COPERNICUS_TOKEN'] = current_cop_token
        else:
            new_token = getpass.getpass("   Enter new Copernicus token: ")
            if new_token.strip():
                credentials['COPERNICUS_TOKEN'] = new_token.strip()
    else:
        cop_token = getpass.getpass("   Enter Copernicus token (or skip): ")
        if cop_token.strip():
            credentials['COPERNICUS_TOKEN'] = cop_token.strip()
    
    # Save to .env file
    if credentials:
        print(f"\n💾 Saving credentials to {env_file}")
        
        # Read existing .env if it exists
        existing_vars = {}
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing_vars[key] = value
        
        # Update with new credentials
        existing_vars.update(credentials)
        
        # Write back to .env
        with open(env_file, 'w') as f:
            f.write("# DEM Downloader Credentials\n")
            f.write("# Generated by setup_credentials.py\n\n")
            for key, value in existing_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"✅ Saved {len(credentials)} credentials")
        
        # Set environment variables for current session
        for key, value in credentials.items():
            os.environ[key] = value
        
    else:
        print("ℹ️  No new credentials provided")
    
    # Summary
    print("\n📊 Credential Summary:")
    print(f"   • OpenTopography: {'✅' if credentials.get('OPENTOPOGRAPHY_API_KEY') else '❌'}")
    print(f"   • NASA Earthdata: {'✅' if credentials.get('EARTHDATA_USER') else '❌'}")
    print(f"   • Copernicus: {'✅' if credentials.get('COPERNICUS_TOKEN') else '❌'}")
    
    available_sources = []
    if credentials.get('OPENTOPOGRAPHY_API_KEY'):
        available_sources.extend(['SRTM GL1', 'SRTM GL3', 'ALOS AW3D30'])
    if credentials.get('EARTHDATA_USER'):
        available_sources.extend(['NASADEM', 'ASTER GDEM'])
    if credentials.get('COPERNICUS_TOKEN'):
        available_sources.extend(['Copernicus GLO-30', 'Copernicus GLO-90'])
    
    print(f"\n🌍 Available DEM sources: {len(available_sources)}")
    for source in available_sources:
        print(f"   • {source}")
    
    if not available_sources:
        print("\n⚠️  WARNING: No credentials configured!")
        print("   The robust downloader will only try sources that don't require authentication.")
    
    return credentials

def test_credentials():
    """Test the configured credentials."""
    print("\n🧪 Testing credentials...")
    
    # Test OpenTopography
    ot_key = os.environ.get('OPENTOPOGRAPHY_API_KEY')
    if ot_key:
        print("   Testing OpenTopography API key...")
        import requests
        try:
            # Test with a tiny request
            url = "https://portal.opentopography.org/API/globaldem"
            params = {
                'demtype': 'SRTMGL1',
                'south': '0.0',
                'north': '0.01',
                'west': '0.0', 
                'east': '0.01',
                'outputFormat': 'GTiff',
                'API_Key': ot_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                print("   ✅ OpenTopography API key valid")
            else:
                print(f"   ❌ OpenTopography error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ OpenTopography test failed: {e}")
    
    # Test NASA Earthdata
    nasa_user = os.environ.get('EARTHDATA_USER')
    if nasa_user:
        print("   ✅ NASA Earthdata credentials configured (login test requires actual download)")
    
    # Test Copernicus
    cop_token = os.environ.get('COPERNICUS_TOKEN')
    if cop_token:
        print("   ✅ Copernicus token configured (validation requires actual API call)")

if __name__ == "__main__":
    print("🚀 DEM Credential Setup")
    print("=" * 30)
    
    setup_credentials()
    test_credentials()
    
    print("\n🎯 Next steps:")
    print("   1. Run: python robust_dem_downloader.py")
    print("   2. Check downloaded DEMs in outputs/robust_dems/")
    print("   3. If needed, run this script again to update credentials")