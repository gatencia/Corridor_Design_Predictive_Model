#!/usr/bin/env python3
"""
GPS Data Visualization Script
Visualize GPS tracks and AOI before running full processing.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Add src to Python path
current_dir = Path.cwd()
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from data_ingestion import GPSDataProcessor

def load_sample_data(data_dir, max_files=5, max_points_per_file=1000):
    """Load a sample of GPS data for quick visualization."""
    
    print(f"📁 Loading sample GPS data from {data_dir}")
    
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        return None, None
    
    print(f"Found {len(csv_files)} files, using first {min(max_files, len(csv_files))} for visualization")
    
    processor = GPSDataProcessor()
    sample_gdfs = []
    
    for i, file_path in enumerate(csv_files[:max_files]):
        try:
            print(f"  Loading {file_path.name}...")
            gdf = processor.load_gps_data(file_path)
            
            # Sample points if file is large
            if len(gdf) > max_points_per_file:
                gdf = gdf.sample(n=max_points_per_file, random_state=42).sort_values('timestamp')
                print(f"    Sampled {max_points_per_file} points from {len(gdf)} total")
            
            sample_gdfs.append(gdf)
            
        except Exception as e:
            print(f"    ❌ Failed to load {file_path.name}: {e}")
            continue
    
    if not sample_gdfs:
        print("❌ No files could be loaded")
        return None, None
    
    # Combine sample data
    combined_gdf = pd.concat(sample_gdfs, ignore_index=True)
    
    # Generate AOI
    print(f"🗺️  Generating AOI from {len(combined_gdf)} GPS points...")
    aoi_gdf = processor.generate_aoi(combined_gdf, buffer_km=5.0)
    
    print(f"✅ Loaded sample data: {len(combined_gdf)} points from {len(sample_gdfs)} files")
    print(f"   AOI area: {aoi_gdf['area_km2'].iloc[0]:.1f} km²")
    
    return combined_gdf, aoi_gdf

def create_overview_map(gdf, aoi_gdf, save_path=None):
    """Create an overview map showing GPS tracks and AOI."""
    
    print("📈 Creating overview map...")
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Convert AOI to geographic coordinates for plotting
    aoi_geo = aoi_gdf.to_crs('EPSG:4326')
    
    # Plot AOI first (as background)
    aoi_geo.plot(ax=ax, color='lightcoral', alpha=0.2, edgecolor='red', 
                linewidth=2, label='AOI (5km buffer)')
    
    # Get unique individuals for different colors
    individuals = gdf['individual-local-identifier'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(individuals)))
    
    print(f"   Plotting tracks for {len(individuals)} individuals...")
    
    # Plot GPS tracks by individual
    for i, individual in enumerate(individuals):
        individual_data = gdf[gdf['individual-local-identifier'] == individual]
        
        # Get collar ID for label
        collar_id = individual_data['tag-local-identifier'].iloc[0]
        
        # Plot points
        individual_data.plot(ax=ax, color=colors[i], markersize=2, alpha=0.7,
                           label=f'Individual {individual} (Collar {collar_id})')
    
    # Customize the plot
    ax.set_title('Cameroon Elephant GPS Tracks and Area of Interest', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    if len(individuals) <= 10:  # Only show legend if not too many individuals
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        # Add a simplified legend
        ax.legend(['AOI (5km buffer)', f'{len(individuals)} GPS collars'], 
                 bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add some stats as text
    bounds = gdf.total_bounds
    stats_text = f"""
    GPS Data Summary:
    • Points: {len(gdf):,}
    • Individuals: {len(individuals)}
    • Date range: {gdf['timestamp'].min().date()} to {gdf['timestamp'].max().date()}
    • Lat range: {bounds[1]:.3f}° to {bounds[3]:.3f}°
    • Lon range: {bounds[0]:.3f}° to {bounds[2]:.3f}°
    • AOI area: {aoi_gdf['area_km2'].iloc[0]:.1f} km²
    """
    
    ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor="white", alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Map saved to: {save_path}")
    
    plt.show()
    
    return fig, ax

def create_detailed_view(gdf, aoi_gdf, individual_id=None, save_path=None):
    """Create a detailed view of a specific individual's tracks."""
    
    if individual_id is None:
        # Use the individual with the most data points
        individual_counts = gdf['individual-local-identifier'].value_counts()
        individual_id = individual_counts.index[0]
    
    individual_data = gdf[gdf['individual-local-identifier'] == individual_id]
    collar_id = individual_data['tag-local-identifier'].iloc[0]
    
    print(f"📈 Creating detailed view for Individual {individual_id} (Collar {collar_id})...")
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Spatial tracks
    aoi_geo = aoi_gdf.to_crs('EPSG:4326')
    aoi_geo.plot(ax=ax1, color='lightblue', alpha=0.2, edgecolor='blue', linewidth=1)
    
    # Plot track as connected line
    coords = [(row['location-long'], row['location-lat']) 
             for _, row in individual_data.iterrows()]
    
    if len(coords) > 1:
        lons, lats = zip(*coords)
        ax1.plot(lons, lats, 'o-', color='red', markersize=3, linewidth=1, 
                alpha=0.7, label='GPS track')
        
        # Mark start and end
        ax1.plot(lons[0], lats[0], 'go', markersize=8, label='Start')
        ax1.plot(lons[-1], lats[-1], 'ro', markersize=8, label='End')
    
    ax1.set_title(f'Individual {individual_id} - Movement Track\n(Collar {collar_id})', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right plot: Temporal movement
    individual_data = individual_data.sort_values('timestamp')
    
    # Calculate daily distances if we have movement data
    if 'distance_km' in individual_data.columns:
        daily_distances = individual_data.groupby(individual_data['timestamp'].dt.date)['distance_km'].sum()
        
        ax2.plot(daily_distances.index, daily_distances.values, 'b-', alpha=0.7)
        ax2.set_title(f'Daily Movement Distance\n(Individual {individual_id})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Distance (km/day)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    else:
        # Just show point count per day
        daily_counts = individual_data.groupby(individual_data['timestamp'].dt.date).size()
        
        ax2.bar(daily_counts.index, daily_counts.values, alpha=0.7, color='green')
        ax2.set_title(f'Daily GPS Fix Count\n(Individual {individual_id})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of GPS fixes')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Detailed view saved to: {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)

def main():
    """Main visualization function."""
    
    print("🐘 GPS Data Visualization Tool")
    print("=" * 60)
    
    # Setup directories
    reports_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GPS data
    data_paths = [
        Path("../GPS_Collar_CSV_Mark"),
        Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
    ]
    
    data_dir = None
    for path in data_paths:
        if path.exists():
            data_dir = path
            break
    
    if not data_dir:
        print("❌ Could not find GPS data directory")
        return
    
    # Load sample data (quick processing)
    gdf, aoi_gdf = load_sample_data(data_dir, max_files=8, max_points_per_file=500)
    
    if gdf is None:
        print("❌ Could not load GPS data")
        return
    
    # Create overview map
    print(f"\n📊 Creating overview map...")
    overview_path = reports_dir / "gps_overview_map.png"
    create_overview_map(gdf, aoi_gdf, save_path=overview_path)
    
    # Create detailed view for individual with most data
    print(f"\n📊 Creating detailed individual view...")
    individual_counts = gdf['individual-local-identifier'].value_counts()
    top_individual = individual_counts.index[0]
    
    detail_path = reports_dir / f"gps_detailed_individual_{top_individual}.png"
    create_detailed_view(gdf, aoi_gdf, individual_id=top_individual, save_path=detail_path)
    
    # Print summary
    print(f"\n📋 Visualization Summary")
    print("=" * 40)
    print(f"Sample data processed: {len(gdf):,} GPS points")
    print(f"Individuals shown: {gdf['individual-local-identifier'].nunique()}")
    print(f"Study sites: {gdf['study-name'].nunique() if 'study-name' in gdf.columns else 'Multiple'}")
    print(f"AOI area: {aoi_gdf['area_km2'].iloc[0]:.1f} km²")
    print(f"Maps saved to: {reports_dir}")
    
    print(f"\n🎯 Ready to run full processing!")
    print("If the maps look good, run: python run_gps_processing.py")

if __name__ == "__main__":
    # Set matplotlib to use a good backend
    import matplotlib
    matplotlib.use('TkAgg')  # Interactive backend
    
    main()