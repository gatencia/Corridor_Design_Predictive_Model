#!/usr/bin/env python3
"""
Main script to run GPS data processing for Cameroon elephant data.
Run this script to process all GPS collar files.
"""

import sys
import os
from pathlib import Path
import logging

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

def main():
    """Main GPS processing function."""
    
    print("🐘 Cameroon Elephant GPS Data Processing")
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
        
        # Process all files
        print(f"\n🔄 Processing {len(gps_files)} GPS files...")
        logger.info("Starting GPS data processing pipeline...")
        
        combined_gdf = processor.process_multiple_files(
            gps_files, 
            output_dir="data/processed"
        )
        
        print(f"✅ Successfully processed {len(combined_gdf):,} GPS fixes")
        print(f"   📊 {combined_gdf['individual-local-identifier'].nunique()} individuals tracked")
        print(f"   📅 Date range: {combined_gdf['timestamp'].min().date()} to {combined_gdf['timestamp'].max().date()}")
        
        # Generate Area of Interest
        print(f"\n🗺️  Generating Area of Interest...")
        logger.info("Generating AOI with 5km buffer...")
        
        aoi_gdf = processor.generate_aoi(
            combined_gdf, 
            buffer_km=5.0, 
            method='convex_hull'
        )
        
        print(f"   📐 AOI area: {aoi_gdf['area_km2'].iloc[0]:.1f} km²")
        print(f"   🌍 Coordinate system: {aoi_gdf['utm_crs'].iloc[0]}")
        
        # Export results
        print(f"\n💾 Exporting results...")
        logger.info("Exporting processed data and AOI...")
        
        processor.export_results(
            combined_gdf, 
            aoi_gdf, 
            output_dir="data/outputs",
            study_name="cameroon_elephants"
        )
        
        print("✅ Results exported successfully!")
        
        # Print detailed summary
        print(f"\n📊 Processing Summary")
        print("=" * 50)
        
        # Data summary
        bounds = combined_gdf.total_bounds
        print(f"GPS Data:")
        print(f"  • Total fixes: {len(combined_gdf):,}")
        print(f"  • Individuals: {combined_gdf['individual-local-identifier'].nunique()}")
        print(f"  • Collar IDs: {combined_gdf['tag-local-identifier'].nunique()}")
        print(f"  • Study sites: {combined_gdf['study-name'].nunique() if 'study-name' in combined_gdf.columns else 'N/A'}")
        print(f"  • Date span: {(combined_gdf['timestamp'].max() - combined_gdf['timestamp'].min()).days} days")
        
        print(f"\nSpatial Extent:")
        print(f"  • Latitude: {bounds[1]:.3f}° to {bounds[3]:.3f}°")
        print(f"  • Longitude: {bounds[0]:.3f}° to {bounds[2]:.3f}°")
        print(f"  • AOI area: {aoi_gdf['area_km2'].iloc[0]:.1f} km²")
        
        # Data quality summary
        if processor.data_quality_report:
            total_valid = sum(report['valid_coordinates'] for report in processor.data_quality_report.values())
            total_fixes = sum(report['total_fixes'] for report in processor.data_quality_report.values())
            total_high_speed = sum(report['high_speed_fixes'] for report in processor.data_quality_report.values())
            total_duplicates = sum(report['duplicates'] for report in processor.data_quality_report.values())
            
            print(f"\nData Quality:")
            print(f"  • Valid coordinates: {total_valid:,}/{total_fixes:,} ({total_valid/total_fixes*100:.1f}%)")
            print(f"  • High-speed fixes flagged: {total_high_speed:,}")
            print(f"  • Duplicates removed: {total_duplicates:,}")
        
        # File outputs
        output_dir = Path("data/outputs")
        output_files = list(output_dir.glob("*cameroon_elephants*"))
        
        print(f"\nOutput Files ({len(output_files)} files created):")
        for file in sorted(output_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  • {file.name} ({size_mb:.1f} MB)")
        
        print(f"\n🎉 GPS data processing completed successfully!")
        print(f"\nNext Steps:")
        print(f"1. 📂 Review outputs in: data/outputs/")
        print(f"2. 📊 Check data quality report: data/processed/data_quality_report_*.json")
        print(f"3. 🗺️  Examine AOI files: data/outputs/aoi_cameroon_elephants_*.geojson")
        print(f"4. ➡️  Proceed to Step 3: EnergyScape implementation")
        
        # Quick visualization if matplotlib available
        if plt is not None:
            print(f"\n📈 Creating quick visualization...")
            create_quick_visualization(combined_gdf, aoi_gdf)
        
        return combined_gdf, aoi_gdf
        
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