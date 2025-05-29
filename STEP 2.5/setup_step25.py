#!/usr/bin/env python3
"""
STEP 2.5 Setup and Validation Script
Verifies environment and sets up STEP 2.5 for DEM downloading.
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime
import importlib.util

def check_python_version():
    """Check if Python version is adequate."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version >= (3, 8):
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (>= 3.8 required)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (>= 3.8 required)")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    # Core dependencies (should already be installed from main project)
    core_deps = [
        ('numpy', 'Numerical computing'),
        ('pandas', 'Data manipulation'),
        ('geopandas', 'Geospatial data processing'),
        ('rasterio', 'Raster data I/O'),
        ('shapely', 'Geometric operations'),
        ('pyproj', 'Coordinate transformations')
    ]
    
    # STEP 2.5 specific dependencies
    step25_deps = [
        ('requests', 'HTTP requests for downloading'),
        ('geopy', 'Geographic distance calculations')
    ]
    
    missing_core = []
    missing_step25 = []
    
    print("   Core dependencies:")
    for pkg, desc in core_deps:
        try:
            __import__(pkg)
            print(f"      ✅ {pkg:<12} - {desc}")
        except ImportError:
            print(f"      ❌ {pkg:<12} - {desc} (MISSING)")
            missing_core.append(pkg)
    
    print("   STEP 2.5 dependencies:")
    for pkg, desc in step25_deps:
        try:
            __import__(pkg)
            print(f"      ✅ {pkg:<12} - {desc}")
        except ImportError:
            print(f"      ❌ {pkg:<12} - {desc} (MISSING)")
            missing_step25.append(pkg)
    
    if missing_core:
        print(f"\n   ⚠️  Missing core dependencies. Install with:")
        print(f"      pip install {' '.join(missing_core)}")
    
    if missing_step25:
        print(f"\n   📥 Installing STEP 2.5 dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_step25)
            print(f"      ✅ STEP 2.5 dependencies installed")
        except subprocess.CalledProcessError:
            print(f"      ❌ Failed to install dependencies. Install manually:")
            print(f"         pip install {' '.join(missing_step25)}")
            return False
    
    return len(missing_core) == 0

def check_project_structure():
    """Check if project structure is correct."""
    print("\n📁 Checking project structure...")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    expected_dirs = [
        project_root / "STEP 1",
        project_root / "STEP 2", 
        project_root / "STEP 3",
        current_dir  # STEP 2.5
    ]
    
    structure_ok = True
    
    for expected_dir in expected_dirs:
        if expected_dir.exists():
            print(f"   ✅ {expected_dir.name}/")
        else:
            print(f"   ❌ {expected_dir.name}/ (missing)")
            structure_ok = False
    
    if not structure_ok:
        print(f"   ⚠️  Project structure issues detected")
        print(f"   Current location: {current_dir}")
        print(f"   Expected project root: {project_root}")
    
    return structure_ok

def check_step2_completion():
    """Check if Step 2 has been completed."""
    print("\n🔍 Checking Step 2 completion...")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Look for Step 2 outputs
    step2_output_paths = [
        project_root / "STEP 2" / "data" / "outputs",
        project_root / "STEP 2" / "outputs",
        project_root / "STEP 2" / "data" / "processed"
    ]
    
    aoi_files_found = 0
    
    for step2_path in step2_output_paths:
        if step2_path.exists():
            # Look for AOI files
            aoi_patterns = ["*aoi*.geojson", "*aoi*.shp", "*AOI*.geojson", "*AOI*.shp"]
            for pattern in aoi_patterns:
                aoi_files_found += len(list(step2_path.rglob(pattern)))
    
    if aoi_files_found > 0:
        print(f"   ✅ Found {aoi_files_found} AOI files from Step 2")
        return True
    else:
        print(f"   ❌ No AOI files found from Step 2")
        print(f"   Complete Step 2 processing before running STEP 2.5")
        
        # Suggest Step 2 execution
        step2_script = project_root / "STEP 2" / "run_gps_processing.py"
        if step2_script.exists():
            print(f"   💡 Run Step 2 with: cd '{step2_script.parent}' && python {step2_script.name}")
        
        return False

def setup_directories():
    """Set up necessary directories for STEP 2.5."""
    print("\n📂 Setting up STEP 2.5 directories...")
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # STEP 2.5 directories
    step25_dirs = [
        current_dir / "logs",
        current_dir / "outputs",
        current_dir / "temp"
    ]
    
    # Step 3 DEM directories (default output location)
    step3_dem_dirs = [
        project_root / "STEP 3" / "data" / "raw" / "dem",
        project_root / "STEP 3" / "data" / "raw" / "dem" / "mosaics",
        project_root / "STEP 3" / "data" / "raw" / "dem" / "tiles",
        project_root / "STEP 3" / "data" / "raw" / "dem" / "metadata"
    ]
    
    all_dirs = step25_dirs + step3_dem_dirs
    
    for directory in all_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory.relative_to(project_root)}")
        except Exception as e:
            print(f"   ❌ {directory.relative_to(project_root)}: {e}")
    
    return True

def test_step25_functionality():
    """Test basic STEP 2.5 functionality."""
    print("\n🧪 Testing STEP 2.5 functionality...")
    
    try:
        # Test imports
        print("   Testing imports...")
        
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        try:
            from aoi_processor import AOIProcessor
            print("      ✅ AOI processor")
        except ImportError as e:
            print(f"      ❌ AOI processor: {e}")
            return False
        
        try:
            from dem_downloader import DEMDownloader
            print("      ✅ DEM downloader")
        except ImportError as e:
            print(f"      ❌ DEM downloader: {e}")
            return False
        
        try:
            from data_organizer import DataOrganizer
            print("      ✅ Data organizer")
        except ImportError as e:
            print(f"      ❌ Data organizer: {e}")
            return False
        
        # Test AOI discovery
        print("   Testing AOI discovery...")
        project_root = current_dir.parent
        processor = AOIProcessor(project_root)
        aois = processor.find_step2_aoi_outputs()
        
        if len(aois) > 0:
            print(f"      ✅ Found {len(aois)} AOIs")
            for aoi in aois[:3]:  # Show first 3
                print(f"         - {aoi['study_site']}: {aoi['area_km2']:.1f} km²")
            if len(aois) > 3:
                print(f"         ... and {len(aois) - 3} more AOIs")
        else:
            print(f"      ⚠️  No AOIs found (Step 2 may not be complete)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

def create_example_config():
    """Create example configuration files."""
    print("\n📝 Creating example configuration...")
    
    current_dir = Path(__file__).parent
    
    # Create example run script
    example_script = current_dir / "run_step25_example.sh"
    
    script_content = """#!/bin/bash
# Example STEP 2.5 execution script

echo "🛰️ STEP 2.5: Automated DEM Download for AOIs"
echo "============================================="

# Check what will be downloaded first
echo "📋 Checking download requirements..."
python download_dem_for_aois.py --dry-run

echo ""
read -p "Proceed with download? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Starting DEM download..."
    python download_dem_for_aois.py --buffer 3.0 --max-concurrent 3
    
    echo ""
    echo "✅ STEP 2.5 complete! Ready for Step 3."
    echo "Next: cd '../STEP 3' && python run_energyscape.py"
else
    echo "Cancelled."
fi
"""
    
    try:
        with open(example_script, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(example_script, 0o755)
        
        print(f"   ✅ Example script: {example_script.name}")
        
        # Create example Python config
        example_py = current_dir / "run_step25_example.py"
        
        py_content = '''#!/usr/bin/env python3
"""
Example STEP 2.5 usage script
Demonstrates programmatic usage of STEP 2.5 components.
"""

from pathlib import Path
from aoi_processor import AOIProcessor
from dem_downloader import DEMDownloader, DEMSource
from data_organizer import DataOrganizer

def main():
    """Example STEP 2.5 workflow."""
    
    print("🛰️ STEP 2.5 Example Workflow")
    print("=" * 40)
    
    # 1. Discover AOIs
    print("\\n1. Discovering AOIs from Step 2...")
    project_root = Path(__file__).parent.parent
    processor = AOIProcessor(project_root)
    aois = processor.find_step2_aoi_outputs()
    
    print(f"   Found {len(aois)} AOIs")
    
    if not aois:
        print("   ❌ No AOIs found. Complete Step 2 first.")
        return
    
    # 2. Calculate requirements
    print("\\n2. Calculating DEM requirements...")
    downloader = DEMDownloader()
    
    total_tiles = set()
    for aoi in aois:
        bounds = aoi['bounds_wgs84']
        buffered_bounds = downloader.buffer_bounds(bounds, 3.0)  # 3km buffer
        tiles = downloader.get_required_dem_tiles(buffered_bounds, DEMSource.NASADEM)
        total_tiles.update(tiles)
    
    print(f"   Need {len(total_tiles)} DEM tiles")
    print(f"   Estimated size: {len(total_tiles) * 25:.0f} MB")
    
    # 3. Show what would be downloaded
    print("\\n3. DEM tiles needed:")
    for i, tile in enumerate(sorted(list(total_tiles))):
        print(f"   {i+1:2d}. {tile}")
        if i >= 4:  # Show first 5
            print(f"   ... and {len(total_tiles) - 5} more tiles")
            break
    
    print("\\n💡 To download: python download_dem_for_aois.py")

if __name__ == "__main__":
    main()
'''
        
        with open(example_py, 'w') as f:
            f.write(py_content)
        
        print(f"   ✅ Example Python script: {example_py.name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating examples: {e}")
        return False

def print_next_steps():
    """Print next steps for user."""
    print("\n🚀 STEP 2.5 Setup Complete!")
    print("=" * 50)
    print()
    print("📋 Next Steps:")
    print("1. Run a dry-run to see what will be downloaded:")
    print("   python download_dem_for_aois.py --dry-run")
    print()
    print("2. Download DEM data with default settings:")
    print("   python download_dem_for_aois.py")
    print()
    print("3. Or use custom buffer and settings:")
    print("   python download_dem_for_aois.py --buffer 5.0 --max-concurrent 3")
    print()
    print("4. After completion, proceed to Step 3:")
    print("   cd '../STEP 3'")
    print("   python run_energyscape.py")
    print()
    print("📚 Documentation:")
    print("   - README.md - Complete usage guide")
    print("   - run_step25_example.py - Example usage")
    print("   - logs/ - Processing logs")
    print()
    print("🆘 Support:")
    print("   - Use --log-level DEBUG for detailed troubleshooting")
    print("   - Check logs/ directory for error details")
    print("   - Ensure Step 2 completed successfully before running")

def main():
    """Main setup function."""
    print("🔧 STEP 2.5 Setup and Validation")
    print("=" * 60)
    print(f"Setup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_success = True
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Step 2 Completion", check_step2_completion),
        ("Directory Setup", setup_directories),
        ("Functionality Test", test_step25_functionality),
        ("Example Configuration", create_example_config)
    ]
    
    for check_name, check_function in checks:
        try:
            result = check_function()
            if not result:
                setup_success = False
        except Exception as e:
            print(f"   ❌ Error in {check_name}: {e}")
            setup_success = False
    
    # Final status
    print(f"\n{'='*60}")
    
    if setup_success:
        print("✅ STEP 2.5 setup completed successfully!")
        print_next_steps()
    else:
        print("❌ STEP 2.5 setup encountered issues")
        print()
        print("🔧 Common fixes:")
        print("   - Install missing dependencies: pip install -r requirements.txt")
        print("   - Complete Step 2 processing first")
        print("   - Check Python version (>= 3.8 required)")
        print("   - Verify project directory structure")
        
    return setup_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)