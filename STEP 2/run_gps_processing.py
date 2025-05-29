#!/usr/bin/env python3
"""
GPS Data Processing: 1 AOI per CSV File
Creates individual AOI for each CSV file regardless of study site.
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

def cleanup_previous_outputs():
    """Clean up previous outputs."""
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
    """Create directory structure."""
    directories = [
        "data/outputs/individual_aois",
        "data/outputs/combined",
        "data/outputs/metadata",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    """Process each CSV file individually - 1 AOI per CSV."""
    
    print("🐘 GPS Data Processing: 1 AOI per CSV File")
    print("=" * 60)
    
    # 1. Cleanup 
    cleanup_previous_outputs()
    
    # 2. Setup
    setup_directories()
    
    # Find GPS data
    data_dir = Path("../GPS_Collar_CSV_Mark")
    if not data_dir.exists():
        data_dir = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
    
    if not data_dir.exists():
        print(f"❌ GPS data directory not found")
        return
    
    gps_files = list(data_dir.glob("*.csv"))
    
    if not gps_files:
        print(f"❌ No CSV files found in {data_dir}")
        return
    
    print(f"📁 Found {len(gps_files)} GPS CSV files")
    print(f"🎯 Will create {len(gps_files)} individual AOIs (1 per CSV)")
    
    try:
        processor = GPSDataProcessor()
        
        print(f"\n🔄 Processing each CSV file individually...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []
        all_aois = []
        
        # Process EVERY CSV file individually
        for i, file_path in enumerate(gps_files, 1):
            print(f"\n📄 Processing {i}/{len(gps_files)}: {file_path.name}")
            
            try:
                # Load GPS data
                gdf = processor.load_gps_data(file_path)
                
                if len(gdf) < 3:
                    print(f"   ⚠️  Skipping - insufficient data points ({len(gdf)})")
                    continue
                
                # Extract study site info and collar details
                filename_parts = file_path.stem.split(" - ")
                if len(filename_parts) >= 2:
                    study_site = filename_parts[1].replace("(Cameroon)", "").replace("(Nigeria)", "").strip()
                    collar_info = filename_parts[-1] if len(filename_parts) > 2 else "Unknown"
                else:
                    study_site = file_path.stem
                    collar_info = "Unknown"
                
                # Create unique identifier for this specific CSV
                unique_id = f"{study_site}_{collar_info}_{i:02d}"
                clean_unique_id = "".join(c for c in unique_id if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
                
                print(f"   📍 Study site: {study_site}")
                print(f"   🏷️  Collar/ID: {collar_info}")
                print(f"   🆔 Unique ID: {clean_unique_id}")
                print(f"   📊 GPS points: {len(gdf)}")
                
                # Generate AOI for THIS specific CSV file
                aoi_gdf = processor.generate_aoi(gdf, buffer_km=5.0, method='convex_hull')
                
                # Add metadata specific to this CSV
                aoi_gdf['study_site'] = study_site
                aoi_gdf['collar_info'] = collar_info
                aoi_gdf['source_file'] = file_path.name
                aoi_gdf['unique_id'] = clean_unique_id
                aoi_gdf['csv_number'] = i
                aoi_gdf['gps_points'] = len(gdf)
                aoi_gdf['tracking_days'] = (gdf['timestamp'].max() - gdf['timestamp'].min()).days
                
                area_km2 = float(aoi_gdf['area_km2'].iloc[0])
                tracking_days = (gdf['timestamp'].max() - gdf['timestamp'].min()).days
                
                print(f"   ✅ AOI created: {area_km2:.1f} km²")
                
                # Create dedicated folder for this AOI
                aoi_base_dir = Path("data/outputs/individual_aois")
                aoi_folder = aoi_base_dir / clean_unique_id
                aoi_folder.mkdir(parents=True, exist_ok=True)
                
                # Save AOI files in dedicated folder
                aoi_geojson = aoi_folder / f"aoi_{clean_unique_id}_{timestamp}.geojson"
                aoi_shapefile = aoi_folder / f"aoi_{clean_unique_id}_{timestamp}.shp"
                metadata_file = aoi_folder / f"metadata_{clean_unique_id}_{timestamp}.json"
                
                aoi_gdf.to_file(aoi_geojson, driver='GeoJSON')
                aoi_gdf.to_file(aoi_shapefile)
                
                print(f"   📁 Created folder: {aoi_folder.name}/")
                print(f"   💾 Saved: {aoi_geojson.name}")
                
                # Store result info
                result_info = {
                    'csv_number': i,
                    'study_site': study_site,
                    'collar_info': collar_info,
                    'unique_id': clean_unique_id,
                    'source_file': file_path.name,
                    'gps_points': len(gdf),
                    'individuals': int(gdf['individual-local-identifier'].nunique()),
                    'area_km2': area_km2,
                    'tracking_days': tracking_days,
                    'aoi_folder': str(aoi_folder),
                    'aoi_geojson': str(aoi_geojson),
                    'aoi_shapefile': str(aoi_shapefile),
                    'metadata_file': str(metadata_file)
                }
                
                # Save metadata in the folder
                with open(metadata_file, 'w') as f:
                    json.dump(result_info, f, indent=2, default=str)
                
                print(f"   📄 Metadata: {metadata_file.name}")
                
                results.append(result_info)
                all_aois.append(aoi_gdf)
                
            except Exception as e:
                print(f"   ❌ Failed to process {file_path.name}: {e}")
                continue
        
        if not results:
            print("❌ No CSV files were successfully processed")
            return
        
        print(f"\n✅ Successfully processed {len(results)} CSV files")
        print(f"📄 Created {len(results)} individual AOIs")
        
        # Create combined AOI file (all AOIs together)
        if all_aois:
            print(f"\n📊 Creating combined AOI file...")
            combined_aoi = pd.concat([aoi.to_crs('EPSG:4326') for aoi in all_aois], ignore_index=True)
            
            combined_dir = Path("data/outputs/combined")
            combined_aoi_file = combined_dir / f"all_aois_combined_{timestamp}.geojson"
            combined_aoi.to_file(combined_aoi_file, driver='GeoJSON')
            print(f"   💾 Combined AOI: {combined_aoi_file.name}")
        
        # Create detailed summary
        summary = {
            'processing_timestamp': timestamp,
            'processing_type': '1_aoi_per_csv_file',
            'total_csv_files': len(gps_files),
            'successfully_processed': len(results),
            'failed_files': len(gps_files) - len(results),
            'total_gps_points': sum(r['gps_points'] for r in results),
            'total_area_km2': sum(r['area_km2'] for r in results),
            'individual_results': results,
            'files_structure': {
                'individual_aois': "data/outputs/individual_aois/{unique_id}/",
                'combined_aoi': "data/outputs/combined/",
                'metadata': "data/outputs/metadata/",
                'folder_organization': "Each AOI has dedicated folder with GeoJSON, Shapefile, and metadata"
            }
        }
        
        summary_file = Path("data/outputs/metadata") / f"processing_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📋 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"CSV files found: {len(gps_files)}")
        print(f"AOIs created: {len(results)}")
        print(f"Total GPS points: {summary['total_gps_points']:,}")
        print(f"Total study area: {summary['total_area_km2']:,.1f} km²")
        
        print(f"\n📄 Individual AOI Folders Created:")
        for result in results:
            folder_name = Path(result['aoi_folder']).name
            print(f"  {result['csv_number']:2d}. {folder_name}/ → {result['area_km2']:,.1f} km² ({result['gps_points']:,} points)")
        
        print(f"\n📁 Organized structure: data/outputs/individual_aois/")
        print(f"   Each AOI has its own dedicated folder with:")
        print(f"   • GeoJSON file")
        print(f"   • Shapefile")  
        print(f"   • Metadata JSON")
        print(f"📄 Summary: {summary_file}")
        
        print(f"\n🚀 Ready for optimized STEP 2.5!")
        print(f"Each CSV file has its own organized folder structure")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()