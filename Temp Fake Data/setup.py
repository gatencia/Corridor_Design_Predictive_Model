#!/usr/bin/env python3
"""
Setup Script for Corridor Prediction Analysis
Ensures all dependencies are installed and properly configured.
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_and_install_package(package_name, import_name=None):
    """Check if package is installed, install if not."""
    
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"⚠️  {package_name} not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def setup_environment():
    """Setup the complete environment for corridor analysis."""
    
    print("🔧 SETTING UP CORRIDOR PREDICTION ANALYSIS ENVIRONMENT")
    print("=" * 60)
    
    # Required packages
    required_packages = [
        ("pandas", "pandas"),
        ("geopandas", "geopandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("rasterio", "rasterio"),
        ("shapely", "shapely"),
        ("scipy", "scipy"),
        ("scikit-image", "skimage"),
        ("pyproj", "pyproj"),
        ("tqdm", "tqdm")
    ]
    
    print("📦 Checking required packages...")
    
    failed_packages = []
    for package_name, import_name in required_packages:
        if not check_and_install_package(package_name, import_name):
            failed_packages.append(package_name)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("Please install these packages manually:")
        for pkg in failed_packages:
            print(f"   pip install {pkg}")
        return False
    
    print(f"\n✅ All required packages are available!")
    
    # Create directory structure
    print(f"\n📁 Setting up directory structure...")
    
    temp_dir = Path("Temp_Fake_Data")
    subdirs = ["rasters", "corridors", "visualizations", "results"]
    
    temp_dir.mkdir(exist_ok=True)
    for subdir in subdirs:
        (temp_dir / subdir).mkdir(exist_ok=True)
        print(f"   Created: {temp_dir / subdir}")
    
    print(f"\n✅ Directory structure ready!")
    
    # Check for Step 2 data
    print(f"\n🔍 Checking for Step 2 data...")
    
    possible_step2_paths = [
        Path("../STEP 2"),
        Path("../STEP_2"),
        Path("STEP 2"),
        Path("STEP_2"),
        Path("../../STEP 2"),
        Path("step2"),
        Path("Step2")
    ]
    
    step2_found = False
    for path in possible_step2_paths:
        if path.exists() and path.is_dir():
            print(f"   ✅ Found Step 2 directory: {path}")
            
            # Check for processed data
            outputs_dir = path / "data" / "outputs"
            if outputs_dir.exists():
                individual_aois = outputs_dir / "individual_aois"
                if individual_aois.exists() and any(individual_aois.iterdir()):
                    print(f"   ✅ Found processed GPS data and AOIs")
                    step2_found = True
                    break
            
            # Check for raw CSV data
            csv_files = list(path.glob("*.csv"))
            if csv_files:
                print(f"   ✅ Found {len(csv_files)} CSV files")
                step2_found = True
                break
    
    if not step2_found:
        print(f"   ⚠️  No Step 2 data found - will use synthetic sample data")
        print(f"   💡 If you have Step 2 data, ensure it's in one of these locations:")
        for path in possible_step2_paths[:3]:
            print(f"      - {path}")
    
    print(f"\n🎯 Environment setup complete!")
    print(f"Ready to run corridor prediction analysis.")
    
    return True

def create_requirements_file():
    """Create requirements.txt file."""
    
    requirements = """# Corridor Prediction Analysis Requirements
pandas>=1.5.0
geopandas>=0.13.0
numpy>=1.24.0
matplotlib>=3.6.0
rasterio>=1.3.0
shapely>=2.0.0
scipy>=1.10.0
scikit-image>=0.20.0
pyproj>=3.6.0
tqdm>=4.65.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("📄 Created requirements.txt file")

def run_quick_test():
    """Run a quick test to ensure everything works."""
    
    print(f"\n🧪 Running Quick Environment Test...")
    
    try:
        import pandas as pd
        import geopandas as gpd
        import numpy as np
        import matplotlib.pyplot as plt
        import rasterio
        from scipy import ndimage
        from skimage import morphology
        
        # Test basic functionality
        print("   ✅ All imports successful")
        
        # Test basic operations
        test_array = np.random.random((10, 10))
        smoothed = ndimage.gaussian_filter(test_array, sigma=1)
        print("   ✅ NumPy and SciPy operations working")
        
        # Test matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.imshow(test_array, cmap='viridis')
        plt.close(fig)
        print("   ✅ Matplotlib working")
        
        # Test geopandas
        from shapely.geometry import Point
        points = [Point(0, 0), Point(1, 1)]
        gdf = gpd.GeoDataFrame({'id': [1, 2]}, geometry=points, crs='EPSG:4326')
        print("   ✅ GeoPandas working")
        
        print(f"\n🎉 Environment test passed! Ready to run analysis.")
        return True
        
    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        return False

def main():
    """Main setup function."""
    
    print("🐘 CORRIDOR PREDICTION ANALYSIS - ENVIRONMENT SETUP")
    print("=" * 70)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed!")
        return False
    
    # Create requirements file
    create_requirements_file()
    
    # Run test
    if not run_quick_test():
        print("\n❌ Environment test failed!")
        return False
    
    print(f"\n🚀 SETUP COMPLETE!")
    print("=" * 30)
    print("Next steps:")
    print("1. Run: python corridor_prediction_analysis.py")
    print("2. Check results in: Temp_Fake_Data/")
    print("3. View visualizations and reports")
    
    return True

if __name__ == "__main__":
    main()