#!/usr/bin/env python3
"""
Quick test script to verify all imports work correctly.
Run this from anywhere in the STEP 2 project.
"""

import sys
import os
from pathlib import Path

def find_project_root():
    """Find the STEP 2 project root directory."""
    current = Path(__file__).parent.absolute()
    
    # Look for src directory going up the tree
    for parent in [current] + list(current.parents):
        src_dir = parent / "src"
        if src_dir.exists() and (src_dir / "data_ingestion.py").exists():
            return parent
    
    return None

def test_imports():
    """Test all imports to make sure everything works."""
    
    print("🧪 Testing GPS Processing Module Imports")
    print("=" * 50)
    
    # Find project root
    project_root = find_project_root()
    if project_root is None:
        print("❌ Could not find project root directory with src/data_ingestion.py")
        print("Make sure you're running this from within the STEP 2 project")
        return False
    
    print(f"📂 Project root: {project_root}")
    
    # Add src to Python path
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Change to project directory for relative imports
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Test 1: Basic imports
        try:
            print("1. Testing data_ingestion import...")
            from data_ingestion import GPSDataProcessor
            print("   ✅ GPSDataProcessor imported successfully")
        except ImportError as e:
            print(f"   ❌ Failed to import GPSDataProcessor: {e}")
            return False
        
        # Test 2: Utils imports
        try:
            print("2. Testing utils imports...")
            from utils.gps_validation import GPSValidator
            from utils.crs_utils import CRSUtils
            from utils.geometry_utils import GeometryUtils
            print("   ✅ All utils imported successfully")
        except ImportError as e:
            print(f"   ❌ Failed to import utils: {e}")
            return False
        
        # Test 3: Exception imports
        try:
            print("3. Testing exception imports...")
            from exceptions.validation_errors import GPSValidationError
            print("   ✅ Exceptions imported successfully")
        except ImportError as e:
            print(f"   ❌ Failed to import exceptions: {e}")
            return False
        
        # Test 4: Required libraries
        print("4. Testing required libraries...")
        missing_libs = []
        
        required_libs = [
            ('pandas', 'pd'),
            ('geopandas', 'gpd'),
            ('numpy', 'np'),
            ('shapely.geometry', 'Point'),
            ('pyproj', None),
            ('geopy.distance', 'geodesic'),
            ('tqdm', None)
        ]
        
        for lib, alias in required_libs:
            try:
                if alias and '.' not in lib:
                    exec(f"import {lib} as {alias}")
                elif alias:
                    exec(f"from {lib} import {alias}")
                else:
                    exec(f"import {lib}")
                print(f"   ✅ {lib}")
            except ImportError:
                print(f"   ❌ {lib} (missing)")
                missing_libs.append(lib)
        
        if missing_libs:
            print(f"\n⚠️  Missing libraries: {', '.join(missing_libs)}")
            print("Install with: pip install pandas geopandas numpy shapely pyproj geopy tqdm")
            return False
        
        # Test 5: Create processor instance
        try:
            print("5. Testing GPSDataProcessor instantiation...")
            processor = GPSDataProcessor()
            print("   ✅ GPSDataProcessor created successfully")
        except Exception as e:
            print(f"   ❌ Failed to create GPSDataProcessor: {e}")
            return False
        
        # Test 6: Check data directory
        print("6. Checking data directory...")
        data_paths = [
            project_root / ".." / "GPS_Collar_CSV_Mark",
            Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
        ]
        
        data_dir = None
        for path in data_paths:
            if path.exists():
                data_dir = path
                break
        
        if data_dir and data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            print(f"   ✅ Data directory found: {data_dir}")
            print(f"   📊 Found {len(csv_files)} CSV files")
            
            # Show a few example files
            if csv_files:
                print(f"   📄 Sample files:")
                for file in sorted(csv_files)[:3]:
                    print(f"      • {file.name}")
                if len(csv_files) > 3:
                    print(f"      ... and {len(csv_files) - 3} more")
        else:
            print(f"   ⚠️  Data directory not found")
            print(f"   Checked paths:")
            for path in data_paths:
                print(f"      • {path}")
            print("   This won't prevent testing, but you'll need to fix the path for processing")
        
        print(f"\n🎉 All import tests passed!")
        print("You're ready to run the GPS processing script.")
        return True
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def show_file_structure():
    """Show the current file structure."""
    print(f"\n📁 Current File Structure:")
    print("=" * 30)
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Find project root
    project_root = find_project_root()
    if project_root is None:
        print("❌ Could not find project root")
        return
    
    print(f"Project root: {project_root}")
    
    src_dir = project_root / "src"
    if src_dir.exists():
        print("\nProject structure:")
        print("src/")
        for file in sorted(src_dir.rglob("*")):
            if file.is_file():
                rel_path = file.relative_to(src_dir)
                print(f"  {rel_path}")
    
    # Check for key files
    key_files = [
        "run_gps_processing.py",
        "src/data_ingestion.py",
        "src/utils/gps_validation.py",
        "src/utils/crs_utils.py", 
        "src/utils/geometry_utils.py",
        "src/exceptions/validation_errors.py"
    ]
    
    print(f"\nKey Files Status:")
    for file_path in key_files:
        file = project_root / file_path
        status = "✅" if file.exists() else "❌"
        print(f"  {status} {file_path}")
        if not file.exists():
            print(f"      Expected at: {file}")

if __name__ == "__main__":
    show_file_structure()
    success = test_imports()
    
    if success:
        print(f"\n🚀 Ready to run GPS processing!")
        print("Next steps:")
        print("1. cd to the STEP 2 directory (if not already there)")
        print("2. python run_gps_processing.py")
    else:
        print(f"\n🔧 Fix the issues above before running GPS processing.")
        
        # Show current location and suggest fix
        current = Path.cwd()
        project_root = find_project_root()
        if project_root:
            print(f"\nSuggested fix:")
            print(f"cd {project_root}")
            print(f"python test_imports.py")