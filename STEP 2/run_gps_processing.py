#!/usr/bin/env python3
"""
Optimized GPS Data Processing for STEP 2
Includes cleanup logic and better file organization.
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime 
import numpy as np
import pandas as pd
import shutil

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from data_ingestion import GPSDataProcessor
    print("✅ Successfully imported GPSDataProcessor")
except ImportError as e:
    print(f"❌ Failed to import GPSDataProcessor: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    plt = None

def cleanup_previous_outputs():
    """Clean up all previous processing outputs to avoid duplicates."""
    print("🧹 Cleaning up previous outputs...")
    
    cleanup_dirs = [
        Path("data/outputs"),
        Path("data/processed"),
        Path("reports")
    ]
    
    for cleanup_dir in cleanup_dirs:
        if cleanup_dir.exists():
            try:
                shutil.rmtree(cleanup_dir)
                print(f"   🗑️  Removed: {cleanup_dir}")
            except Exception as e:
                print(f"   ⚠️  Could not remove {cleanup_dir}: {e}")
    
    print("✅ Cleanup complete")

def setup_directories():
    """Create organized directory structure."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/outputs/individual_aois",
        "data/outputs/combined_aois",
        "data/outputs/metadata",
        "logs",
        "reports/figures"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def setup_logging():
    """Setup logging configuration."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Clean old log files (keep only last 5)
    log_files = sorted(logs_dir.glob("gps_processing_*.log"))
    if len(log_files) > 5:
        for old_log in log_files[:-5]:
            old_log.unlink()
    
    log_file = logs_dir / f'gps_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_study_site_folders(individual_results, timestamp):
    """Create organized folders for each study site."""
    base_dir = Path("data/outputs/individual_aois")
    
    # Group by study site
    sites_data = {}
    for result in individual_results:
        site_name = result['study_site']
        if site_name not in sites_data:
            sites_data[site_name] = []
        sites_data[site_name].append(result)
    
    # Create folders and organize files
    organized_results = []
    
    for site_name, site_results in sites_data.items():
        # Create clean folder name
        clean_name = "".join(c for c in site_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        site_dir = base_dir / clean_name
        site_dir.mkdir(exist_ok=True)
        
        # Take the most recent result for this site (or combine if multiple collars)
        if len(site_results) == 1:
            main_result = site_results[0]
        else:
            # Combine multiple collars for same site
            main_result = {
                'study_site': site_name,
                'collar_info': f"{len(site_results)} collars",
                'source_files': [r['source_file'] for r in site_results],
                'gps_points': sum(r['gps_points'] for r in site_results),
                'individuals': sum(r['individuals'] for r in site_results),
                'area_km2': max(r['area_km2'] for r in site_results),  # Use largest AOI
                'tracking_days': max(r['tracking_days'] for r in site_results)
            }
        
        # Create single AOI file for this site
        site_aoi_name = f"aoi_{clean_name}_{timestamp}"
        
        main_result['organized_files'] = {
            'site_directory': str(site_dir),
            'aoi_geojson': str(site_dir / f"{site_aoi_name}.geojson"),
            'aoi_shapefile': str(site_dir / f"{site_aoi_name}.shp"),
            'metadata': str(site_dir / f"site_metadata_{timestamp}.json")
        }
        
        organized_results.append(main_result)
    
    return organized_results

def main():
    """Optimized main GPS processing function."""
    
    print("🐘 OPTIMIZED Cameroon Elephant GPS Data Processing")
    print("=" * 60)
    
    # 1. Cleanup previous outputs first
    cleanup_previous_outputs()
    
    # 2. Setup
    setup_directories()
    logger = setup_logging()
    
    # Find GPS data
    data_dir = Path("../GPS_Collar_CSV_Mark")
    if not data_dir.exists():
        data_dir = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
    
    if not data_dir.exists():
        logger.error(f"GPS data directory not found")
        return
    
    gps_files = list(data_dir.glob("*.csv"))
    
    if not gps_files:
        logger.error(f"No CSV files found in {data_dir}")
        return
    
    print(f"📁 Found {len(gps_files)} GPS CSV files")
    
    try:
        processor = GPSDataProcessor()
        
        # Process files to identify unique study sites
        print(f"\n🔄 Processing GPS files to identify unique study sites...")
        
        individual_results = []
        all_aois = []
        combined_gps_data = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process each file but group by study site
        study_sites_processed = set()
        
        for i, file_path in enumerate(gps_files, 1):
            print(f"\n📄 Processing {i}/{len(gps_files)}: {file_path.name}")
            
            try:
                gdf = processor.load_gps_data(file_path)
                
                if len(gdf) < 3:
                    print(f"   ⚠️  Skipping - insufficient data points ({len(gdf)})")
                    continue
                
                # Extract study site info
                filename_parts = file_path.stem.split(" - ")
                if len(filename_parts) >= 2:
                    study_site = filename_parts[1].replace("(Cameroon)", "").replace("(Nigeria)", "").strip()
                    collar_info = filename_parts[-1] if len(filename_parts) > 2 else "Unknown"
                else:
                    study_site = file_path.stem
                    collar_info = "Unknown"
                
                # Skip if we've already processed this study site (avoid duplicates)
                site_key = study_site.lower().replace(" ", "").replace("_", "")
                if site_key in study_sites_processed:
                    print(f"   ℹ️  Study site '{study_site}' already processed, skipping duplicate")
                    continue
                
                study_sites_processed.add(site_key)
                
                # Generate AOI for this study site
                aoi_gdf = processor.generate_aoi(gdf, buffer_km=5.0, method='convex_hull')
                
                # Add metadata
                aoi_gdf['study_site'] = study_site
                aoi_gdf['collar_info'] = collar_info
                aoi_gdf['source_file'] = file_path.name
                aoi_gdf['gps_points'] = len(gdf)
                aoi_gdf['tracking_days'] = (gdf['timestamp'].max() - gdf['timestamp'].min()).days
                
                result_info = {
                    'study_site': study_site,
                    'collar_info': collar_info,
                    'source_file': file_path.name,
                    'gps_points': len(gdf),
                    'individuals': int(gdf['individual-local-identifier'].nunique()),
                    'area_km2': float(aoi_gdf['area_km2'].iloc[0]),
                    'tracking_days': (gdf['timestamp'].max() - gdf['timestamp'].min()).days
                }
                
                individual_results.append(result_info)
                all_aois.append(aoi_gdf)
                combined_gps_data.append(gdf)
                
                print(f"   ✅ {study_site}: {len(gdf)} points → {aoi_gdf['area_km2'].iloc[0]:.1f} km² AOI")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        if not individual_results:
            print("❌ No unique study sites were successfully processed")
            return
        
        print(f"\n✅ Processed {len(individual_results)} unique study sites")
        
        # Create organized folder structure
        organized_results = create_study_site_folders(individual_results, timestamp)
        
        # Save individual AOI files in organized structure
        for i, (result, aoi_gdf) in enumerate(zip(organized_results, all_aois)):
            try:
                files = result['organized_files']
                
                # Save to organized location
                aoi_gdf.to_file(files['aoi_geojson'], driver='GeoJSON')
                aoi_gdf.to_file(files['aoi_shapefile'])
                
                # Save metadata
                with open(files['metadata'], 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                print(f"   📁 Organized: {result['study_site']} → {Path(files['site_directory']).name}/")
                
            except Exception as e:
                logger.error(f"Failed to organize {result['study_site']}: {e}")
        
        # Create combined files
        print(f"\n📊 Creating combined datasets...")
        
        if len(combined_gps_data) > 0:
            combined_gdf = pd.concat(combined_gps_data, ignore_index=True)
            combined_gdf = combined_gdf.sort_values(['individual-local-identifier', 'timestamp'])
            combined_gdf = combined_gdf.drop_duplicates(
                subset=['individual-local-identifier', 'timestamp', 'location-lat', 'location-long'], 
                keep='first'
            )
        
        if len(all_aois) > 0:
            combined_aoi = pd.concat([aoi.to_crs('EPSG:4326') for aoi in all_aois], ignore_index=True)
            
            # Save combined AOI (single file)
            combined_dir = Path("data/outputs/combined_aois")
            combined_aoi_file = combined_dir / f"all_study_sites_combined_{timestamp}.geojson"
            combined_aoi.to_file(combined_aoi_file, driver='GeoJSON')
        
        # Export combined results
        if len(combined_gps_data) > 0 and len(all_aois) > 0:
            processor.export_results(
                combined_gdf, 
                combined_aoi.iloc[[0]],
                output_dir="data/outputs/combined_aois",
                study_name="cameroon_elephants_optimized"
            )
        
        # Create processing summary
        summary = {
            'processing_timestamp': timestamp,
            'unique_study_sites_processed': len(organized_results),
            'total_gps_points': sum(r['gps_points'] for r in organized_results),
            'total_area_km2': sum(r['area_km2'] for r in organized_results),
            'optimization_notes': [
                "Duplicates removed during processing",
                "Files organized by study site",
                "Single AOI per study site",
                "Ready for efficient STEP 2.5 processing"
            ],
            'study_sites': organized_results,
            'files_structure': {
                'individual_aois': "data/outputs/individual_aois/{site_name}/",
                'combined_aois': "data/outputs/combined_aois/",
                'metadata': "data/outputs/metadata/"
            }
        }
        
        summary_file = Path("data/outputs/metadata") / f"processing_summary_{timestamp}.json"
        summary_file.parent.mkdir(exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("✅ Optimized GPS processing completed!")
        
        # Print organized summary
        print(f"\n📊 OPTIMIZED Processing Summary")
        print("=" * 50)
        print(f"Unique study sites: {len(organized_results)}")
        print(f"Total GPS fixes: {summary['total_gps_points']:,}")
        print(f"Total study area: {summary['total_area_km2']:,.1f} km²")
        print(f"Organized structure: data/outputs/individual_aois/")
        
        print(f"\nStudy Sites (Deduplicated):")
        for result in sorted(organized_results, key=lambda x: x['area_km2'], reverse=True):
            print(f"  📍 {result['study_site']}: {result['area_km2']:,.1f} km² ({result['gps_points']:,} points)")
        
        print(f"\n🚀 Ready for optimized STEP 2.5 processing!")
        print(f"Each study site has dedicated folder with single AOI file")
        
        return organized_results
        
    except Exception as e:
        logger.error(f"GPS processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()