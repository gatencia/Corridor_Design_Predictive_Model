#!/usr/bin/env python3
"""
Main script to run GPS data processing for Cameroon elephant data.
Run this script to process all GPS collar files.
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime 
import numpy as np


import pandas as pd

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import after adding to path
try:
    from data_ingestion import GPSDataProcessor
    print("✅ Successfully imported GPSDataProcessor")
except ImportError as e:
    print(f"❌ Failed to import GPSDataProcessor: {e}")
    print("Make sure all files are in the correct locations")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    print("⚠️ Matplotlib not available, skipping visualization")
    plt = None

# Configure logging
def setup_logging():
    """Setup logging configuration."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'gps_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/outputs",
        "logs",
        "reports/figures"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def create_individual_aoi_visualization(combined_aoi, individual_results):
    """Create enhanced visualization of individual AOIs with GPS points and background map."""
    if plt is None:
        return
        
    try:
        import contextily as ctx
        has_contextily = True
    except ImportError:
        print("⚠️  contextily not available - will create map without background tiles")
        has_contextily = False
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Convert AOIs to Web Mercator for background map compatibility
        aoi_mercator = combined_aoi.to_crs('EPSG:3857')  # Web Mercator
        
        # Get unique study sites for colors
        study_sites = aoi_mercator['study_site'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(study_sites)))
        
        # Plot each AOI with semi-transparent fill
        for i, site in enumerate(study_sites):
            site_data = aoi_mercator[aoi_mercator['study_site'] == site]
            site_data.plot(ax=ax, color=colors[i], alpha=0.3, 
                          edgecolor='black', linewidth=2, label=f'{site} AOI')
        
        # Load and plot GPS points for each study site
        print("   Adding GPS points to visualization...")
        
        # Find GPS data files to load points
        data_paths = [
            Path("../GPS_Collar_CSV_Mark"),
            Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
        ]
        
        data_dir = None
        for path in data_paths:
            if path.exists():
                data_dir = path
                break
        
        if data_dir:
            # Load GPS points for visualization (sample for performance)
            processor = GPSDataProcessor()
            
            for i, result in enumerate(individual_results):
                try:
                    # Find the corresponding GPS file
                    gps_file = data_dir / result['source_file']
                    if gps_file.exists():
                        # Load GPS data
                        gdf = processor.load_gps_data(gps_file)
                        
                        # Sample points for performance (max 200 points per site)
                        if len(gdf) > 200:
                            gdf_sample = gdf.sample(n=200, random_state=42)
                        else:
                            gdf_sample = gdf
                        
                        # Convert to Web Mercator
                        gdf_mercator = gdf_sample.to_crs('EPSG:3857')
                        
                        # Plot GPS points
                        gdf_mercator.plot(ax=ax, color=colors[i % len(colors)], 
                                        markersize=8, alpha=0.7, marker='o',
                                        edgecolors='white', linewidth=0.5)
                        
                except Exception as e:
                    print(f"   Could not load GPS points for {result['study_site']}: {e}")
                    continue
        
        # Add background map if contextily is available
        if has_contextily:
            try:
                print("   Adding background map...")
                ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.CartoDB.Positron, 
                               alpha=0.7, zoom='auto')
            except Exception as e:
                print(f"   Could not add background map: {e}")
        
        # Customize the plot
        ax.set_title('Cameroon Elephant Study Sites - Individual AOIs with GPS Tracks', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Remove axis ticks for cleaner look with background map
        ax.tick_params(labelsize=8)
        
        # Add legend with better positioning
        legend_elements = []
        for i, site in enumerate(study_sites):
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                           markerfacecolor=colors[i], markersize=10,
                                           alpha=0.7, label=f'{site}'))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                 loc='upper left', fontsize=10, framealpha=0.9)
        
        # Add summary statistics with better formatting
        total_area = sum(r['area_km2'] for r in individual_results)
        total_points = sum(r['gps_points'] for r in individual_results)
        
        stats_text = f"""Study Site Summary:
- Sites: {len(individual_results)}
- Total GPS points: {total_points:,}
- Total AOI area: {total_area:,.0f} km²
- Countries: Cameroon & Nigeria
- Date range: 2003-2024"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor="white", alpha=0.9, edgecolor='gray'), 
               fontsize=11, fontweight='bold')
        
        # Add north arrow and scale (simple version)
        ax.text(0.95, 0.05, '↑ N', transform=ax.transAxes, 
               fontsize=16, fontweight='bold', ha='center',
               bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save with high resolution
        output_file = Path("reports/figures/individual_aois_with_gps_and_basemap.png")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Enhanced AOI visualization saved: {output_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not create enhanced AOI visualization: {e}")
        # Fallback to simple visualization
        create_simple_aoi_visualization(combined_aoi, individual_results)

def create_simple_aoi_visualization(combined_aoi, individual_results):
    """Simple fallback visualization without background map."""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Convert to geographic coordinates
        aoi_geo = combined_aoi.to_crs('EPSG:4326')
        
        # Get unique study sites for colors
        study_sites = aoi_geo['study_site'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(study_sites)))
        
        # Plot each AOI
        for i, site in enumerate(study_sites):
            site_data = aoi_geo[aoi_geo['study_site'] == site]
            site_data.plot(ax=ax, color=colors[i], alpha=0.4, 
                          edgecolor='black', linewidth=1, label=site)
        
        ax.set_title('Cameroon Elephant Study Sites - Individual AOIs', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        output_file = Path("reports/figures/individual_aois_simple.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Simple AOI visualization saved: {output_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not create simple visualization: {e}")


def main():
    """Main GPS processing function with individual AOI generation."""
    
    print("🐘 Cameroon Elephant GPS Data Processing - Individual AOIs")
    print("=" * 60)
    
    # Setup
    setup_directories()
    logger = setup_logging()
    
    # Updated path to match your structure
    data_dir = Path("../GPS_Collar_CSV_Mark")  # Relative to STEP 2 directory
    
    # If relative path doesn't work, try absolute path
    if not data_dir.exists():
        data_dir = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
    
    if not data_dir.exists():
        logger.error(f"GPS data directory not found: {data_dir}")
        print(f"❌ Data directory not found: {data_dir}")
        print("Please check the path and try again.")
        print(f"Current working directory: {Path.cwd()}")
        return
    
    # Find all CSV files
    gps_files = list(data_dir.glob("*.csv"))
    
    if not gps_files:
        logger.error(f"No CSV files found in {data_dir}")
        print(f"❌ No CSV files found in {data_dir}")
        return
    
    print(f"📁 Found {len(gps_files)} GPS CSV files:")
    for i, file in enumerate(gps_files[:10], 1):  # Show first 10 files
        print(f"   {i:2d}. {file.name}")
    if len(gps_files) > 10:
        print(f"   ... and {len(gps_files) - 10} more files")
    
    try:
        # Initialize processor
        logger.info("Initializing GPS data processor...")
        processor = GPSDataProcessor()
        
        # Process files individually to create separate AOIs
        print(f"\n🔄 Processing {len(gps_files)} GPS files individually...")
        logger.info("Starting individual GPS data processing pipeline...")
        
        individual_results = []
        all_aois = []
        combined_gps_data = []
        
        # Create individual AOIs directory
        aoi_dir = Path("data/outputs/individual_aois")
        aoi_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process each file individually
        for i, file_path in enumerate(gps_files, 1):
            print(f"\n📄 Processing {i}/{len(gps_files)}: {file_path.name}")
            
            try:
                # Load GPS data for this file
                gdf = processor.load_gps_data(file_path)
                
                if len(gdf) < 3:
                    print(f"   ⚠️  Skipping - insufficient data points ({len(gdf)})")
                    continue
                
                # Extract study site info from filename
                filename_parts = file_path.stem.split(" - ")
                if len(filename_parts) >= 2:
                    study_site = filename_parts[1].replace("(Cameroon)", "").strip()
                    collar_info = filename_parts[-1] if len(filename_parts) > 2 else "Unknown"
                else:
                    study_site = file_path.stem
                    collar_info = "Unknown"
                
                # Generate AOI for this individual dataset
                aoi_gdf = processor.generate_aoi(gdf, buffer_km=5.0, method='convex_hull')
                
                # Add metadata to AOI
                aoi_gdf['study_site'] = study_site
                aoi_gdf['collar_info'] = collar_info
                aoi_gdf['source_file'] = file_path.name
                aoi_gdf['gps_points'] = len(gdf)
                aoi_gdf['tracking_days'] = (gdf['timestamp'].max() - gdf['timestamp'].min()).days
                
                # Save individual AOI files
                clean_name = study_site.replace(" ", "_").replace("/", "_")
                aoi_geojson = aoi_dir / f"aoi_{clean_name}_{timestamp}.geojson"
                aoi_shp = aoi_dir / f"aoi_{clean_name}_{timestamp}.shp"
                
                aoi_gdf.to_file(aoi_geojson, driver='GeoJSON')
                aoi_gdf.to_file(aoi_shp)
                
                # Store results
                result_info = {
                    'study_site': study_site,
                    'collar_info': collar_info,
                    'source_file': file_path.name,
                    'gps_points': len(gdf),
                    'individuals': int(gdf['individual-local-identifier'].nunique()),
                    'area_km2': float(aoi_gdf['area_km2'].iloc[0]),
                    'tracking_days': (gdf['timestamp'].max() - gdf['timestamp'].min()).days,
                    'files_created': {
                        'geojson': str(aoi_geojson),
                        'shapefile': str(aoi_shp)
                    }
                }
                
                individual_results.append(result_info)
                all_aois.append(aoi_gdf)
                combined_gps_data.append(gdf)
                
                print(f"   ✅ {study_site}: {len(gdf)} points → {aoi_gdf['area_km2'].iloc[0]:.1f} km² AOI")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                print(f"   ❌ Failed to process {file_path.name}: {e}")
                continue
        
        if not individual_results:
            print("❌ No AOIs were successfully created")
            return
        
        # Combine all GPS data for overall statistics
        logger.info("Combining GPS data for overall statistics...")
        combined_gdf = pd.concat(combined_gps_data, ignore_index=True)
        combined_gdf = combined_gdf.sort_values(['individual-local-identifier', 'timestamp'])
        
        # Remove duplicates across files
        duplicate_cols = ['individual-local-identifier', 'timestamp', 'location-lat', 'location-long']
        combined_gdf = combined_gdf.drop_duplicates(subset=duplicate_cols, keep='first')
        
        # Create combined AOI file
        print(f"\n📊 Creating combined AOI file...")
        print(f"   Converting {len(all_aois)} AOIs to common CRS...")
        aois_wgs84 = []
        for aoi in all_aois:
            aoi_wgs84 = aoi.to_crs('EPSG:4326')  # Convert to WGS84
            aois_wgs84.append(aoi_wgs84)

        combined_aoi = pd.concat(aois_wgs84, ignore_index=True)

        
        combined_aoi_geojson = Path("data/outputs") / f"all_aois_combined_{timestamp}.geojson"
        combined_aoi_shp = Path("data/outputs") / f"all_aois_combined_{timestamp}.shp"
        
        combined_aoi.to_file(combined_aoi_geojson, driver='GeoJSON')
        combined_aoi.to_file(combined_aoi_shp)
        
        # Export combined GPS data
        processor.export_results(
            combined_gdf, 
            combined_aoi.iloc[[0]],  # Use first AOI for format compatibility
            output_dir="data/outputs",
            study_name="cameroon_elephants_combined"
        )
        
        # Save individual AOI summary report
        summary_report = {
            'processing_timestamp': timestamp,
            'total_study_sites': len(individual_results),
            'buffer_distance_km': 5.0,
            'total_gps_points': sum(r['gps_points'] for r in individual_results),
            'total_aoi_area_km2': sum(r['area_km2'] for r in individual_results),
            'study_sites': individual_results,
            'files_created': {
                'combined_aoi_geojson': str(combined_aoi_geojson),
                'combined_aoi_shapefile': str(combined_aoi_shp),
                'individual_aois_directory': str(aoi_dir)
            }
        }
        
        summary_file = Path("data/outputs") / f"individual_aoi_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print("✅ Individual AOI processing completed successfully!")
        
        # Print detailed summary
        print(f"\n📊 Individual AOI Processing Summary")
        print("=" * 50)
        
        # Overall statistics
        total_points = sum(r['gps_points'] for r in individual_results)
        total_area = sum(r['area_km2'] for r in individual_results)
        
        print(f"Study Sites Processed: {len(individual_results)}")
        print(f"Total GPS fixes: {total_points:,}")
        print(f"Combined GPS fixes (deduplicated): {len(combined_gdf):,}")
        print(f"Individuals tracked: {combined_gdf['individual-local-identifier'].nunique()}")
        print(f"Date range: {combined_gdf['timestamp'].min().date()} to {combined_gdf['timestamp'].max().date()}")
        print(f"Total AOI area: {total_area:,.1f} km²")
        
        # Individual site details
        print(f"\nIndividual Study Sites:")
        sorted_results = sorted(individual_results, key=lambda x: x['area_km2'], reverse=True)
        for i, result in enumerate(sorted_results, 1):
            print(f"  {i:2d}. {result['study_site']}")
            print(f"      GPS points: {result['gps_points']:,}")
            print(f"      AOI area: {result['area_km2']:,.1f} km²")
            print(f"      Tracking days: {result['tracking_days']}")
        
        # File outputs
        print(f"\nOutput Files:")
        print(f"  📁 Individual AOIs: {aoi_dir}")
        print(f"  🗺️  Combined AOI: {combined_aoi_geojson}")
        print(f"  📊 Summary report: {summary_file}")
        
        print(f"\n🎉 Individual AOI processing completed successfully!")
        print(f"\nNext Steps:")
        print(f"1. 📂 Review individual AOI files in: {aoi_dir}")
        print(f"2. 🗺️  Examine combined AOI: {combined_aoi_geojson}")
        print(f"3. 📊 Check summary report: {summary_file}")
        print(f"4. ➡️  Use individual AOIs for corridor analysis between study sites")
        
        # Quick visualization if matplotlib available
        if plt is not None:
            print(f"\n📈 Creating visualization...")
            create_individual_aoi_visualization(combined_aoi, individual_results)
        
        return individual_results, combined_aoi
        
    except Exception as e:
        logger.error(f"GPS processing failed: {e}")
        print(f"\n❌ GPS processing failed: {e}")
        print(f"Check the log file: logs/gps_processing.log")
        import traceback
        traceback.print_exc()
        raise


def create_quick_visualization(gdf, aoi_gdf):
    """Create a quick visualization of the results."""
    if plt is None:
        return
        
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Map of GPS tracks and AOI
        aoi_geo = aoi_gdf.to_crs('EPSG:4326')
        aoi_geo.plot(ax=ax1, color='lightcoral', alpha=0.3, edgecolor='red', linewidth=2)
        
        # Plot GPS tracks by individual (sample for performance)
        if len(gdf) > 10000:
            gdf_sample = gdf.sample(n=10000, random_state=42)
        else:
            gdf_sample = gdf
            
        individuals = gdf_sample['individual-local-identifier'].unique()
        colors = plt.cm.Set1(range(len(individuals)))
        
        for i, individual in enumerate(individuals):
            individual_data = gdf_sample[gdf_sample['individual-local-identifier'] == individual]
            individual_data.plot(ax=ax1, color=colors[i], markersize=0.5, alpha=0.6, 
                               label=f'Individual {individual}')
        
        ax1.set_title('GPS Tracks and AOI')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        if len(individuals) <= 10:  # Only show legend if not too many individuals
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Daily GPS fix count
        gdf['date'] = gdf['timestamp'].dt.date
        daily_counts = gdf.groupby('date').size()
        daily_counts.plot(ax=ax2, kind='line', alpha=0.7)
        ax2.set_title('Daily GPS Fix Count Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of GPS Fixes')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = Path("reports/figures/gps_processing_summary.png")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {output_file}")
        
        plt.close()  # Close the figure to free memory
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

if __name__ == "__main__":
    main()