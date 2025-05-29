#!/usr/bin/env python3
"""
STEP 2.5: Automated DEM Download for AOIs
Downloads DEM data for areas of interest generated in Step 2.
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

# Import components
try:
    from dem_downloader import DEMDownloader, DEMSource
    from data_organizer import DataOrganizer
    from aoi_processor import AOIProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all STEP 2.5 components are in the same directory")
    sys.exit(1)

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
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
    logger.info(f"STEP 2.5 logging initialized - log file: {log_file}")
    return logger

def discover_step2_aois() -> List[Dict[str, Any]]:
    """Discover AOI files from Step 2 outputs."""
    print("🔍 Discovering Step 2 AOI outputs...")
    
    # Use the existing AOI processor
    aoi_processor = AOIProcessor(project_root)
    aoi_files = aoi_processor.find_step2_aoi_outputs()
    
    if not aoi_files:
        print("❌ No AOI files found from Step 2")
        print("Make sure Step 2 processing has been completed")
        return []
    
    print(f"✅ Found {len(aoi_files)} AOI files:")
    for aoi in aoi_files:
        print(f"   • {aoi['study_site']}: {aoi['area_km2']:.1f} km² ({aoi['format']})")
    
    return aoi_files

def calculate_dem_requirements(aoi_files: List[Dict[str, Any]], 
                             buffer_km: float = 2.0) -> Dict[str, Any]:
    """Calculate DEM tile requirements for all AOIs."""
    print(f"\n📐 Calculating DEM requirements (buffer: {buffer_km} km)...")
    
    downloader = DEMDownloader()
    requirements = {
        'total_aois': len(aoi_files),
        'dem_tiles_needed': set(),
        'total_coverage_bounds': None,
        'estimated_size_mb': 0,
        'aoi_details': []
    }
    
    all_bounds = []
    
    for aoi in aoi_files:
        # Get buffered bounds
        bounds = aoi['bounds']
        buffered_bounds = downloader.buffer_bounds(bounds, buffer_km)
        
        # Calculate required tiles
        tiles = downloader.get_required_dem_tiles(buffered_bounds, DEMSource.NASADEM)
        
        aoi_detail = {
            'study_site': aoi['study_site'],
            'original_bounds': bounds,
            'buffered_bounds': buffered_bounds,
            'required_tiles': list(tiles),
            'tile_count': len(tiles)
        }
        
        requirements['aoi_details'].append(aoi_detail)
        requirements['dem_tiles_needed'].update(tiles)
        all_bounds.extend(buffered_bounds)
        
        print(f"   • {aoi['study_site']}: {len(tiles)} tiles needed")
    
    # Calculate overall coverage bounds
    if all_bounds:
        min_lon = min(all_bounds[0::4])
        min_lat = min(all_bounds[1::4])
        max_lon = max(all_bounds[2::4])
        max_lat = max(all_bounds[3::4])
        requirements['total_coverage_bounds'] = (min_lon, min_lat, max_lon, max_lat)
    
    # Estimate download size
    tile_count = len(requirements['dem_tiles_needed'])
    requirements['estimated_size_mb'] = tile_count * 25  # ~25MB per NASADEM tile
    
    print(f"📊 DEM Requirements Summary:")
    print(f"   Total unique tiles needed: {tile_count}")
    print(f"   Estimated download size: {requirements['estimated_size_mb']:.0f} MB")
    print(f"   Coverage area: {requirements['total_coverage_bounds']}")
    
    return requirements

def download_dem_data(requirements: Dict[str, Any], 
                     output_dir: Path,
                     source: DEMSource = DEMSource.NASADEM,
                     max_concurrent: int = 3) -> Dict[str, Any]:
    """Download required DEM data."""
    print(f"\n📥 Starting DEM download ({source.name})...")
    
    downloader = DEMDownloader(
        output_dir=output_dir,
        max_concurrent=max_concurrent
    )
    
    tiles_to_download = list(requirements['dem_tiles_needed'])
    
    if not tiles_to_download:
        print("⚠️ No tiles to download")
        return {'success': True, 'downloaded_files': [], 'errors': []}
    
    print(f"📋 Download queue: {len(tiles_to_download)} tiles")
    
    # Download tiles
    download_results = downloader.download_dem_tiles(tiles_to_download, source)
    
    # Summary
    successful = len(download_results['downloaded_files'])
    failed = len(download_results['errors'])
    
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {successful}/{len(tiles_to_download)} tiles")
    print(f"   ❌ Failed: {failed} tiles")
    
    if download_results['errors']:
        print(f"   Errors:")
        for error in download_results['errors'][:3]:  # Show first 3 errors
            print(f"     • {error}")
        if len(download_results['errors']) > 3:
            print(f"     ... and {len(download_results['errors']) - 3} more errors")
    
    # Calculate actual download size
    total_size_mb = sum(f['size_mb'] for f in download_results['downloaded_files'])
    print(f"   💾 Total downloaded: {total_size_mb:.1f} MB")
    
    return download_results

def organize_and_prepare_data(download_results: Dict[str, Any],
                            aoi_files: List[Dict[str, Any]],
                            output_dir: Path) -> Dict[str, Any]:
    """Organize downloaded data for Step 3 consumption."""
    print(f"\n📁 Organizing data for Step 3...")
    
    organizer = DataOrganizer(output_dir)
    
    # Create mosaics for each AOI
    organization_results = {
        'mosaics_created': [],
        'individual_tiles_organized': len(download_results['downloaded_files']),
        'errors': []
    }
    
    for aoi in aoi_files:
        try:
            print(f"   Processing {aoi['study_site']}...")
            
            # Find tiles that cover this AOI
            relevant_tiles = []
            for file_info in download_results['downloaded_files']:
                tile_path = Path(file_info['path'])
                # Simple check - could be more sophisticated
                if tile_path.exists():
                    relevant_tiles.append(tile_path)
            
            if relevant_tiles:
                # Create mosaic for this AOI
                mosaic_info = organizer.create_aoi_mosaic(
                    dem_tiles=relevant_tiles,
                    aoi_bounds=aoi['bounds'],
                    aoi_name=aoi['study_site'],
                    buffer_km=2.0
                )
                
                if mosaic_info:
                    organization_results['mosaics_created'].append(mosaic_info)
                    print(f"     ✅ Created mosaic: {mosaic_info['output_path'].name}")
                else:
                    error_msg = f"Failed to create mosaic for {aoi['study_site']}"
                    organization_results['errors'].append(error_msg)
                    print(f"     ❌ {error_msg}")
            else:
                error_msg = f"No tiles found for {aoi['study_site']}"
                organization_results['errors'].append(error_msg)
                print(f"     ⚠️ {error_msg}")
                
        except Exception as e:
            error_msg = f"Error processing {aoi['study_site']}: {e}"
            organization_results['errors'].append(error_msg)
            print(f"     ❌ {error_msg}")
    
    # Organize tile library
    organizer.organize_tile_library(download_results['downloaded_files'])
    
    # Create metadata
    metadata = organizer.create_metadata_summary(
        aoi_files, download_results, organization_results
    )
    
    print(f"\n📊 Organization Summary:")
    print(f"   ✅ AOI mosaics created: {len(organization_results['mosaics_created'])}")
    print(f"   📦 Individual tiles organized: {organization_results['individual_tiles_organized']}")
    print(f"   📄 Metadata file: {metadata['metadata_path']}")
    
    return organization_results

def validate_step3_readiness(output_dir: Path) -> bool:
    """Validate that data is ready for Step 3 consumption."""
    print(f"\n🔍 Validating Step 3 readiness...")
    
    # Check expected directory structure
    expected_dirs = [
        output_dir / "mosaics",
        output_dir / "tiles" / "nasadem",
        output_dir / "metadata"
    ]
    
    validation_issues = []
    
    for expected_dir in expected_dirs:
        if not expected_dir.exists():
            validation_issues.append(f"Missing directory: {expected_dir}")
        else:
            print(f"   ✅ {expected_dir.name}/")
    
    # Check for DEM files
    dem_files = list(output_dir.rglob("*.tif"))
    if len(dem_files) == 0:
        validation_issues.append("No DEM files found")
    else:
        print(f"   ✅ Found {len(dem_files)} DEM files")
    
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
        print(f"   ✅ All checks passed - ready for Step 3")
        return True

def main():
    """Main STEP 2.5 function."""
    parser = argparse.ArgumentParser(
        description="STEP 2.5: Download DEM data for Step 2 AOIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_dem_for_aois.py                    # Download with defaults
  python download_dem_for_aois.py --buffer 5.0       # Use 5km buffer
  python download_dem_for_aois.py --output-dir /path/to/dem --max-concurrent 5
  python download_dem_for_aois.py --dry-run          # Show what would be downloaded
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
                       help='Show what would be downloaded without downloading')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    print("🛰️ STEP 2.5: Automated DEM Download for AOIs")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Discover Step 2 AOIs
        aoi_files = discover_step2_aois()
        if not aoi_files:
            print("❌ Cannot proceed without AOI files from Step 2")
            sys.exit(1)
        
        # 2. Calculate DEM requirements
        source = DEMSource.NASADEM if args.source == 'nasadem' else DEMSource.SRTM
        requirements = calculate_dem_requirements(aoi_files, args.buffer)
        
        if args.dry_run:
            print(f"\n🔍 DRY RUN - Would download:")
            print(f"   📦 {len(requirements['dem_tiles_needed'])} tiles")
            print(f"   💾 ~{requirements['estimated_size_mb']:.0f} MB")
            print(f"   📍 Tiles: {sorted(list(requirements['dem_tiles_needed']))}")
            return
        
        # 3. Download DEM data
        download_results = download_dem_data(
            requirements, output_dir, source, args.max_concurrent
        )
        
        if not download_results['downloaded_files']:
            print("❌ No DEM files were downloaded successfully")
            sys.exit(1)
        
        # 4. Organize and prepare data
        organization_results = organize_and_prepare_data(
            download_results, aoi_files, output_dir
        )
        
        # 5. Validate Step 3 readiness
        ready_for_step3 = validate_step3_readiness(output_dir)
        
        # Final summary
        print(f"\n🎉 STEP 2.5 Processing Complete!")
        print("=" * 50)
        print(f"✅ AOIs processed: {len(aoi_files)}")
        print(f"✅ DEM tiles downloaded: {len(download_results['downloaded_files'])}")
        print(f"✅ Mosaics created: {len(organization_results['mosaics_created'])}")
        print(f"📁 Output directory: {output_dir}")
        
        if ready_for_step3:
            print(f"\n🚀 Ready for Step 3!")
            print(f"   Run: cd '{step3_dir}' && python run_energyscape.py")
        else:
            print(f"\n⚠️ Issues found - check validation messages above")
        
        # Save processing summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'aois_processed': len(aoi_files),
            'dem_source': source.name,
            'buffer_km': args.buffer,
            'tiles_downloaded': len(download_results['downloaded_files']),
            'mosaics_created': len(organization_results['mosaics_created']),
            'output_directory': str(output_dir),
            'ready_for_step3': ready_for_step3,
            'aoi_details': requirements['aoi_details']
        }
        
        summary_file = output_dir / "step25_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📄 Processing summary saved: {summary_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during processing")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    
    finally:
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()