# Main EnergyScape processing script
#!/usr/bin/env python3
"""
Main script to run EnergyScape energy landscape calculations.
Processes DEM data and generates energy cost surfaces for elephant corridor analysis.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import argparse
import json

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Setup logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"energyscape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - log file: {log_file}")
    return logger

def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    print("🔍 Checking EnergyScape dependencies...")
    
    required_packages = [
        'numpy', 'rasterio', 'geopandas', 'shapely', 'pyproj'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} (missing)")
    
    # Check R integration
    try:
        from energyscape.r_integration import EnerscapeRBridge
        r_bridge = EnerscapeRBridge()
        if r_bridge.r_available:
            print(f"   ✅ R integration available")
            if r_bridge.enerscape_available:
                print(f"   ✅ R enerscape package available")
            else:
                print(f"   ⚠️  R enerscape package not available (will use Python fallback)")
        else:
            print(f"   ⚠️  R not available (will use Python implementation)")
    except ImportError as e:
        print(f"   ⚠️  R integration import failed: {e}")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available")
    return True

def discover_step2_outputs() -> dict:
    """Discover and validate Step 2 outputs."""
    print("\n📁 Discovering Step 2 outputs...")
    
    # Try different possible Step 2 locations
    step2_paths = [
        Path("../STEP 2/data/outputs"),
        Path("../STEP 2/outputs"),
        current_dir.parent / "STEP 2" / "data" / "outputs"
    ]
    
    step2_info = {
        'found': False,
        'directory': None,
        'aoi_files': [],
        'gps_files': []
    }
    
    for step2_path in step2_paths:
        if step2_path.exists():
            step2_info['found'] = True
            step2_info['directory'] = step2_path
            
            # Find AOI files
            aoi_files = list(step2_path.rglob("*aoi*.geojson")) + list(step2_path.rglob("*aoi*.shp"))
            step2_info['aoi_files'] = [str(f) for f in aoi_files]
            
            # Find GPS files
            gps_files = list(step2_path.rglob("*gps*.gpq")) + list(step2_path.rglob("*gps*.csv"))
            step2_info['gps_files'] = [str(f) for f in gps_files]
            
            print(f"   ✅ Step 2 outputs found: {step2_path}")
            print(f"   📐 AOI files: {len(aoi_files)}")
            print(f"   📍 GPS files: {len(gps_files)}")
            break
    
    if not step2_info['found']:
        print("   ❌ Step 2 outputs not found in expected locations")
        print("   Make sure Step 2 has been completed and outputs are available")
    
    return step2_info

def find_dem_data() -> dict:
    """Find available DEM data."""
    print("\n🗻 Searching for DEM data...")
    
    dem_search_dirs = [
        # Step 2.5 robust DEM downloader outputs (PRIORITY)
        Path("../STEP 2.5/outputs/robust_dems"),
        Path("../STEP 2.5/outputs/aoi_dems"), 
        Path("../STEP 2.5/outputs/aoi_specific_dems"),
        # Traditional locations
        Path("data/raw/dem"),
        Path("../data/raw/dem"),
        Path("/tmp/dem")  # Common download location
    ]
    
    dem_info = {
        'found': False,
        'directories': [],
        'files': []
    }
    
    for dem_dir in dem_search_dirs:
        if dem_dir.exists():
            dem_files = []
            for ext in ['.tif', '.tiff', '.img', '.hgt', '.bil', '.asc']:
                dem_files.extend(dem_dir.rglob(f'*{ext}'))
                dem_files.extend(dem_dir.rglob(f'*{ext.upper()}'))
            
            if dem_files:
                dem_info['found'] = True
                dem_info['directories'].append(str(dem_dir))
                dem_info['files'].extend([str(f) for f in dem_files])
                print(f"   ✅ DEM files found in: {dem_dir}")
                print(f"   📊 Files: {len(dem_files)}")
                for f in dem_files[:3]:  # Show first 3 files
                    print(f"     • {f.name}")
                if len(dem_files) > 3:
                    print(f"     ... and {len(dem_files) - 3} more files")
    
    if not dem_info['found']:
        print("   ⚠️  No DEM files found in search directories")
        print("   You may need to download DEM data for your study area")
        print("   Suggested sources: NASADEM, SRTM, or ASTER GDEM")
        print("   Or run Step 2.5: cd '../STEP 2.5' && python robust_dem_downloader.py")
    
    return dem_info

def run_energy_calculations(args) -> dict:
    """Run the main energy landscape calculations."""
    print("\n⚡ Starting EnergyScape energy landscape calculations...")
    print("=" * 70)
    
    try:
        # Import EnergyScape components
        from energyscape.core import EnergyScapeProcessor
        from config.energyscape_config import get_energyscape_manager
        
        # Initialize processor
        config_file = args.config if hasattr(args, 'config') and args.config else None
        processor = EnergyScapeProcessor(config_file=config_file)
        
        # Print configuration summary
        if processor.config:
            try:
                if hasattr(processor, 'config_manager') and processor.config_manager:
                    processor.config_manager.print_summary()
                else:
                    print("⚡ EnergyScape Configuration Summary")
                    print("=" * 60)
                    print("Using default configuration parameters")
            except AttributeError:
                print("⚡ EnergyScape processor initialized with default configuration")
        
        # Discover Step 2 AOIs
        aoi_files = processor.process_step2_aoi()
        
        if not aoi_files:
            print("❌ No AOI files found from Step 2")
            print("Please ensure Step 2 has been completed successfully")
            return {'success': False, 'error': 'No AOI files found'}
        
        print(f"📐 Found {len(aoi_files)} AOI files to process")
        
        # Setup DEM search directories
        dem_search_dirs = None
        if hasattr(args, 'dem_dir') and args.dem_dir:
            dem_search_dirs = [Path(args.dem_dir)]
        
        # Setup output directory
        output_dir = None
        if hasattr(args, 'output_dir') and args.output_dir:
            output_dir = Path(args.output_dir)
        
        # Run processing
        results = processor.process_all_aois(
            dem_search_dirs=dem_search_dirs,
            output_dir=output_dir
        )
        
        # Print results summary
        print(f"\n📊 EnergyScape Processing Results:")
        print(f"   Total AOIs processed: {results['total_aois']}")
        print(f"   Successful: {results['successful_aois']}")
        print(f"   Failed: {results['failed_aois']}")
        print(f"   Energy surfaces created: {len(results['energy_surfaces_created'])}")
        
        if results['errors']:
            print(f"   Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"     • {error}")
            if len(results['errors']) > 3:
                print(f"     ... and {len(results['errors']) - 3} more errors")
        
        # Validate results if any surfaces were created
        if results['energy_surfaces_created']:
            print(f"\n🔍 Validating energy surfaces...")
            surface_paths = [s['path'] for s in results['energy_surfaces_created']]
            validation = processor.validate_energy_surfaces(surface_paths)
            
            print(f"   Valid surfaces: {validation['valid_surfaces']}/{validation['total_surfaces']}")
            
            if validation['invalid_surfaces'] > 0:
                print(f"   ⚠️  {validation['invalid_surfaces']} surfaces have validation issues")
        
        return results
        
    except ImportError as e:
        error_msg = f"Failed to import EnergyScape components: {e}"
        print(f"❌ {error_msg}")
        return {'success': False, 'error': error_msg}
    except Exception as e:
        error_msg = f"Error during energy calculations: {e}"
        print(f"❌ {error_msg}")
        return {'success': False, 'error': error_msg}

def install_r_packages() -> bool:
    """Install required R packages."""
    print("\n📦 Installing R packages for EnergyScape...")
    
    try:
        from energyscape.r_integration import install_r_dependencies
        success = install_r_dependencies()
        
        if success:
            print("✅ R packages installed successfully")
        else:
            print("❌ Failed to install R packages")
            print("   You can still use the Python implementation")
        
        return success
        
    except Exception as e:
        print(f"❌ Error installing R packages: {e}")
        return False

def create_sample_config() -> None:
    """Create sample configuration file."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    sample_config = {
        'energyscape': {
            'pontzer_alpha': 1.0,
            'pontzer_beta': 0.75,
            'pontzer_gamma': 1.5,
            'elephant_mass_female': 2744.0,
            'elephant_mass_male': 6029.0,
            'dem_resolution_m': 30.0,
            'use_r_enerscape': True,
            'output_format': 'GTiff',
            'max_workers': 4
        }
    }
    
    config_file = config_dir / "energyscape_sample.yml"
    
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print(f"📝 Sample configuration created: {config_file}")

def main():
    """Main EnergyScape processing function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="EnergyScape: Generate energy landscape surfaces for elephant corridor analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_energyscape.py                    # Run with default settings
  python run_energyscape.py --check            # Check dependencies only
  python run_energyscape.py --install-r        # Install R packages
  python run_energyscape.py --dem-dir /path/to/dem --output-dir /path/to/output
  python run_energyscape.py --config config/custom.yml
        """
    )
    
    parser.add_argument('--check', action='store_true',
                       help='Check dependencies and Step 2 outputs only')
    parser.add_argument('--install-r', action='store_true',
                       help='Install required R packages')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration file')
    parser.add_argument('--config', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--dem-dir', type=str,
                       help='Directory containing DEM files')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for energy surfaces')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    print("⚡ EnergyScape Implementation - Energy Landscape Calculations")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Handle special commands
    if args.create_config:
        create_sample_config()
        return
    
    if args.install_r:
        install_r_packages()
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        print("Please install missing packages and try again")
        sys.exit(1)
    
    # Discover Step 2 outputs
    step2_info = discover_step2_outputs()
    
    # Find DEM data
    dem_info = find_dem_data()
    
    # Check-only mode
    if args.check:
        print(f"\n✅ Dependency and data check completed")
        print(f"Step 2 outputs: {'✅ Found' if step2_info['found'] else '❌ Not found'}")
        print(f"DEM data: {'✅ Found' if dem_info['found'] else '⚠️  Not found'}")
        return
    
    # Validate prerequisites
    if not step2_info['found']:
        print("\n❌ Cannot proceed without Step 2 outputs")
        print("Please complete Step 2 data processing first")
        sys.exit(1)
    
    if not dem_info['found']:
        print("\n⚠️  No DEM data found - you may need to provide DEM files")
        print("Use --dem-dir to specify DEM location, or download DEM data")
        
        # Ask user if they want to continue anyway
        try:
            response = input("Continue anyway? (y/N): ").lower().strip()
            if response != 'y':
                print("Exiting...")
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(1)
    
    # Run the main energy calculations
    try:
        results = run_energy_calculations(args)
        
        if results.get('success', True):
            print(f"\n🎉 EnergyScape processing completed successfully!")
            
            if results.get('energy_surfaces_created'):
                print(f"\n📁 Energy surfaces created:")
                for surface in results['energy_surfaces_created'][:5]:  # Show first 5
                    print(f"   • {Path(surface['path']).name}")
                    print(f"     {surface['study_site']} - {surface['elephant_sex']} ({surface['body_mass_kg']} kg)")
                
                if len(results['energy_surfaces_created']) > 5:
                    print(f"   ... and {len(results['energy_surfaces_created']) - 5} more surfaces")
            
            print(f"\n🔗 Next Steps:")
            print(f"   1. Review generated energy surfaces in the output directory")
            print(f"   2. Proceed to Step 4: Corridor Analysis using these energy surfaces")
            print(f"   3. Validate results against known elephant movement patterns")
            
        else:
            print(f"\n❌ EnergyScape processing failed")
            if 'error' in results:
                print(f"Error: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        print("Partial results may be available in the output directory")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during processing")
        print(f"\n❌ Unexpected error: {e}")
        print("Check the log file for detailed error information")
        sys.exit(1)
    
    finally:
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()