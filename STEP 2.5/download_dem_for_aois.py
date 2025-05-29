#!/usr/bin/env python3
"""
OPTIMIZED STEP 2.5: Smart DEM Download for Unique AOIs
Downloads DEM data efficiently with deduplication and optimization.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import argparse
import json
from typing import List, Dict, Any, Optional

# Add project paths
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
step2_dir = project_root / "STEP 2"
step3_dir = project_root / "STEP 3"

sys.path.extend([str(step2_dir / "src"), str(step3_dir / "src")])

# Import optimized components
try:
    from dem_downloader import DEMDownloader, DEMSource
    from data_organizer import DataOrganizer
    from aoi_processor import OptimizedAOIProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all STEP 2.5 components are in the same directory")
    sys.exit(1)

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup optimized logging."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Clean old logs (keep only last 5)
    log_files = sorted(logs_dir.glob("step25_dem_download_*.log"))
    if len(log_files) > 5:
        for old_log in log_files[:-5]:
            old_log.unlink()
    
    log_file = logs_dir / f"step25_dem_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"OPTIMIZED STEP 2.5 logging initialized - log file: {log_file}")
    return logger

def discover_unique_aois() -> List[Dict[str, Any]]:
    """Discover unique AOI files with smart deduplication."""
    print("🔍 Discovering unique Step 2 AOI outputs (with deduplication)...")
    
    processor = OptimizedAOIProcessor(project_root)
    aoi_files = processor.find_step2_aoi_outputs()
    
    if not aoi_files:
        print("❌ No unique AOI files found from Step 2")
        print("Make sure Step 2 processing has been completed with the optimized version")
        return []
    
    print(f"✅ Found {len(aoi_files)} unique AOI files (duplicates removed):")
    
    # Sort by area for display
    sorted_aois = sorted(aoi_files, key=lambda x: x['area_km2'], reverse=True)
    
    for aoi in sorted_aois:
        print(f"   📍 {aoi['study_site']}: {aoi['area_km2']:.1f} km² ({aoi['format']})")
    
    return aoi_files

def calculate_optimized_requirements(aoi_files: List[Dict[str, Any]], 
                                   buffer_km: float = 2.0) -> Dict[str, Any]:
    """Calculate DEM requirements with overlap optimization."""
    print(f"\n📐 Calculating optimized DEM requirements (buffer: {buffer_km} km)...")
    
    processor = OptimizedAOIProcessor(project_root)
    requirements = processor.calculate_unique_dem_requirements(aoi_files, buffer_km)
    
    print(f"📊 Optimization Results:")
    print(f"   Unique AOIs: {requirements['unique_aois']}")
    print(f"   Individual tile calculations: {requirements['total_individual_tiles']}")
    print(f"   Unique tiles needed: {requirements['unique_tiles_needed']}")
    print(f"   🚀 Efficiency gain: {requirements['efficiency_gain_percent']:.1f}% reduction")
    print(f"   💾 Estimated download: {requirements['estimated_size_mb']:.0f} MB")
    
    return requirements

def download_dem_data_optimized(requirements: Dict[str, Any], 
                              output_dir: Path,
                              source: DEMSource = DEMSource.NASADEM,
                              max_concurrent: int = 3) -> Dict[str, Any]:
    """Download DEM data with optimizations."""
    print(f"\n📥 Starting optimized DEM download ({source.name})...")
    
    downloader = DEMDownloader(
        output_dir=output_dir,
        max_concurrent=max_concurrent
    )
    
    unique_tiles = requirements['tile_list']
    
    if not unique_tiles:
        print("⚠️ No tiles to download")
        return {'success': True, 'downloaded_files': [], 'errors': []}
    
    print(f"📋 Optimized download queue: {len(unique_tiles)} unique tiles")
    print(f"💡 Efficiency: Downloading {len(unique_tiles)} instead of {requirements['total_individual_tiles']} tiles")
    
    # Download unique tiles only
    download_results = downloader.download_dem_tiles(unique_tiles, source)
    
    # Enhanced summary
    successful = len(download_results['downloaded_files'])
    failed = len(download_results['errors'])
    
    print(f"\n📊 Optimized Download Summary:")
    print(f"   ✅ Successful: {successful}/{len(unique_tiles)} tiles")
    print(f"   ❌ Failed: {failed} tiles")
    print(f"   🚀 Tiles saved by optimization: {requirements['total_individual_tiles'] - len(unique_tiles)}")
    
    if download_results['errors']:
        print(f"   Errors:")
        for error in download_results['errors'][:3]:
            print(f"     • {error}")
        if len(download_results['errors']) > 3:
            print(f"     ... and {len(download_results['errors']) - 3} more errors")
    
    total_size_mb = sum(f['size_mb'] for f in download_results['downloaded_files'])
    print(f"   💾 Total downloaded: {total_size_mb:.1f} MB")
    
    return download_results

def organize_and_prepare_data_optimized(download_results: Dict[str, Any],
                                      aoi_files: List[Dict[str, Any]],
                                      output_dir: Path,
                                      buffer_km: float = 2.0) -> Dict[str, Any]:
    """Organize downloaded data efficiently."""
    print(f"\n📁 Organizing data for Step 3 (optimized)...")
    
    organizer = DataOrganizer(output_dir)
    
    organization_results = {
        'mosaics_created': [],
        'individual_tiles_organized': len(download_results['downloaded_files']),
        'errors': []
    }
    
    # Create mosaics for each unique AOI
    for aoi in aoi_files:
        try:
            print(f"   Processing {aoi['study_site']}...")
            
            # Get all downloaded tiles (they're all available since we downloaded unique set)
            relevant_tiles = [Path(file_info['path']) for file_info in download_results['downloaded_files']]
            
            if relevant_tiles:
                # Create mosaic for this AOI
                mosaic_info = organizer.create_aoi_mosaic(
                    dem_tiles=relevant_tiles,
                    aoi_bounds=aoi['bounds'],
                    aoi_name=aoi['study_site'],
                    buffer_km=buffer_km
                )
                
                if mosaic_info:
                    organization_results['mosaics_created'].append(mosaic_info)
                    print(f"     ✅ Created mosaic: {mosaic_info['output_path'].name}")
                else:
                    error_msg = f"Failed to create mosaic for {aoi['study_site']}"
                    organization_results['errors'].append(error_msg)
                    print(f"     ❌ {error_msg}")
            else:
                error_msg = f"No tiles available for {aoi['study_site']}"
                organization_results['errors'].append(error_msg)
                print(f"     ⚠️ {error_msg}")
                
        except Exception as e:
            error_msg = f"Error processing {aoi['study_site']}: {e}"
            organization_results['errors'].append(error_msg)
            print(f"     ❌ {error_msg}")
    
    # Organize tile library
    organizer.organize_tile_library(download_results['downloaded_files'])
    
    # Create enhanced metadata
    metadata = organizer.create_metadata_summary(
        aoi_files, download_results, organization_results
    )
    
    print(f"\n📊 Organization Summary:")
    print(f"   ✅ AOI mosaics created: {len(organization_results['mosaics_created'])}")
    print(f"   📦 Individual tiles organized: {organization_results['individual_tiles_organized']}")
    print(f"   📄 Metadata file: {metadata['metadata_path']}")
    
    return organization_results

def validate_step3_readiness_optimized(output_dir: Path, aoi_count: int) -> bool:
    """Enhanced Step 3 readiness validation."""
    print(f"\n🔍 Validating Step 3 readiness (optimized)...")
    
    validation_issues = []
    
    # Check directory structure
    expected_dirs = [
        output_dir / "mosaics",
        output_dir / "tiles" / "nasadem", 
        output_dir / "metadata"
    ]
    
    for expected_dir in expected_dirs:
        if not expected_dir.exists():
            validation_issues.append(f"Missing directory: {expected_dir}")
        else:
            print(f"   ✅ {expected_dir.name}/")
    
    # Check DEM mosaics (should match AOI count)
    mosaic_files = list((output_dir / "mosaics").glob("*.tif"))
    if len(mosaic_files) == 0:
        validation_issues.append("No DEM mosaic files found")
    elif len(mosaic_files) < aoi_count:
        validation_issues.append(f"Only {len(mosaic_files)} mosaics found, expected {aoi_count}")
    else:
        print(f"   ✅ Found {len(mosaic_files)} DEM mosaics (matches {aoi_count} AOIs)")
    
    # Check tile library
    tile_files = list((output_dir / "tiles").rglob("*.zip"))
    if not tile_files:
        validation_issues.append("No DEM tiles in library")
    else:
        print(f"   ✅ Found {len(tile_files)} DEM tiles in library")
    
    # Check metadata
    metadata_files = list((output_dir / "metadata").glob("*.json"))
    if not metadata_files:
        validation_issues.append("No metadata files found")
    else:
        print(f"   ✅ Found {len(metadata_files)} metadata files")
    
    if validation_issues:
        print(f"   ❌ Validation issues:")
        for issue in validation_issues:
            print(f"     • {issue}")
        return False
    else:
        print(f"   ✅ All checks passed - optimized data ready for Step 3")
        return True

def main():
    """Optimized main STEP 2.5 function."""
    parser = argparse.ArgumentParser(
        description="OPTIMIZED STEP 2.5: Smart DEM download for unique AOIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimizations:
  - Automatic deduplication of AOI files
  - Smart tile overlap detection
  - Organized file structure support
  - Reduced download redundancy
  
Examples:
  python download_dem_for_aois.py                    # Optimized download
  python download_dem_for_aois.py --buffer 5.0       # Larger buffer
  python download_dem_for_aois.py --dry-run          # Show optimization benefits
        """
    )
    
    parser.add_argument('--output-dir', type=str,
                       default=str(step3_dir / "data" / "raw" / "dem"),
                       help='Output directory for DEM data')
    parser.add_argument('--buffer', type=float, default=2.0,
                       help='Buffer around AOIs in kilometers')
    parser.add_argument('--source', type=str, default='nasadem',
                       choices=['nasadem', 'srtm'],
                       help='DEM data source')
    parser.add_argument('--max-concurrent', type=int, default=3,
                       help='Maximum concurrent downloads')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show optimization benefits without downloading')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    print("🚀 OPTIMIZED STEP 2.5: Smart DEM Download for Unique AOIs")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Optimizations: Deduplication ✅ | Tile Overlap Detection ✅ | Smart Organization ✅")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Discover unique AOIs (with deduplication)
        aoi_files = discover_unique_aois()
        if not aoi_files:
            print("❌ Cannot proceed without unique AOI files from Step 2")
            print("💡 Try running the optimized Step 2 processing first")
            sys.exit(1)
        
        # 2. Calculate optimized DEM requirements
        source = DEMSource.NASADEM if args.source == 'nasadem' else DEMSource.SRTM
        requirements = calculate_optimized_requirements(aoi_files, args.buffer)
        
        if args.dry_run:
            print(f"\n🔍 DRY RUN - Optimization Benefits:")
            print(f"   📦 Unique tiles to download: {requirements['unique_tiles_needed']}")
            print(f"   📦 Without optimization: {requirements['total_individual_tiles']} tiles")
            print(f"   🚀 Efficiency gain: {requirements['efficiency_gain_percent']:.1f}% reduction")
            print(f"   💾 Download size: ~{requirements['estimated_size_mb']:.0f} MB")
            print(f"   ⏱️  Estimated time saved: {(requirements['total_individual_tiles'] - requirements['unique_tiles_needed']) * 30 / 60:.1f} minutes")
            
            print(f"\n📍 Unique study sites to be processed:")
            for aoi in sorted(aoi_files, key=lambda x: x['area_km2'], reverse=True):
                print(f"     • {aoi['study_site']}: {aoi['area_km2']:.1f} km²")
            
            return
        
        # 3. Download DEM data (optimized)
        download_results = download_dem_data_optimized(
            requirements, output_dir, source, args.max_concurrent
        )
        
        if not download_results['downloaded_files']:
            print("❌ No DEM files were downloaded successfully")
            sys.exit(1)
        
        # 4. Organize and prepare data
        organization_results = organize_and_prepare_data_optimized(
            download_results, aoi_files, output_dir, args.buffer
        )
        
        # 5. Validate Step 3 readiness
        ready_for_step3 = validate_step3_readiness_optimized(output_dir, len(aoi_files))
        
        # Final summary with optimization metrics
        print(f"\n🎉 OPTIMIZED STEP 2.5 Processing Complete!")
        print("=" * 60)
        print(f"✅ Unique AOIs processed: {len(aoi_files)}")
        print(f"✅ DEM tiles downloaded: {len(download_results['downloaded_files'])}")
        print(f"✅ Mosaics created: {len(organization_results['mosaics_created'])}")
        print(f"🚀 Tiles saved by optimization: {requirements['total_individual_tiles'] - requirements['unique_tiles_needed']}")
        print(f"⏱️  Time saved: ~{(requirements['total_individual_tiles'] - requirements['unique_tiles_needed']) * 30 / 60:.1f} minutes")
        print(f"📁 Output directory: {output_dir}")
        
        if ready_for_step3:
            print(f"\n🚀 Ready for Step 3!")
            print(f"   Run: cd '{step3_dir}' && python run_energyscape.py")
        else:
            print(f"\n⚠️ Issues found - check validation messages above")
        
        # Save optimization summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'optimization_enabled': True,
            'unique_aois_processed': len(aoi_files),
            'dem_source': source.name,
            'buffer_km': args.buffer,
            'tiles_downloaded': len(download_results['downloaded_files']),
            'tiles_saved_by_optimization': requirements['total_individual_tiles'] - requirements['unique_tiles_needed'],
            'efficiency_gain_percent': requirements['efficiency_gain_percent'],
            'mosaics_created': len(organization_results['mosaics_created']),
            'output_directory': str(output_dir),
            'ready_for_step3': ready_for_step3,
            'unique_study_sites': [aoi['study_site'] for aoi in aoi_files]
        }
        
        summary_file = output_dir / "step25_optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📄 Optimization summary saved: {summary_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during optimized processing")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    
    finally:
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()