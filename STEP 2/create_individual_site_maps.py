#!/usr/bin/env python3
"""
Create Individual Site Maps and Compile to PDF
Generates one clear map per GPS collar/study site showing AOI, GPS tracks, and background map.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
current_dir = Path.cwd()
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from data_ingestion import GPSDataProcessor

def create_individual_site_map(gps_file_path, output_dir, site_number, total_sites):
    """
    Create a detailed map for a single GPS collar/study site.
    
    Parameters:
    -----------
    gps_file_path : Path
        Path to the GPS CSV file
    output_dir : Path
        Directory to save the map
    site_number : int
        Current site number (for progress)
    total_sites : int
        Total number of sites
    """
    
    print(f"📄 Creating map {site_number}/{total_sites}: {gps_file_path.name}")
    
    try:
        # Check for contextily (background maps)
        try:
            import contextily as ctx
            has_contextily = True
        except ImportError:
            has_contextily = False
            print(f"   ⚠️  No background map available (install contextily)")
        
        # Initialize processor and load data
        processor = GPSDataProcessor()
        gdf = processor.load_gps_data(gps_file_path)
        
        if len(gdf) < 3:
            print(f"   ⚠️  Skipping - insufficient data points ({len(gdf)})")
            return None
        
        # Generate AOI
        aoi_gdf = processor.generate_aoi(gdf, buffer_km=5.0, method='convex_hull')
        
        # Extract site information
        filename_parts = gps_file_path.stem.split(" - ")
        if len(filename_parts) >= 2:
            study_site = filename_parts[1].replace("(Cameroon)", "").replace("(Nigeria)", "").strip()
            collar_info = filename_parts[-1] if len(filename_parts) > 2 else "Unknown Collar"
        else:
            study_site = gps_file_path.stem
            collar_info = "Unknown Collar"
        
        # Set up the figure with proper size for PDF
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))  # Letter size
        
        # Convert to Web Mercator for background map compatibility
        if has_contextily:
            gdf_plot = gdf.to_crs('EPSG:3857')
            aoi_plot = aoi_gdf.to_crs('EPSG:3857')
        else:
            gdf_plot = gdf
            aoi_plot = aoi_gdf.to_crs('EPSG:4326')
        
        # Plot AOI first (background)
        aoi_plot.plot(ax=ax, color='lightcoral', alpha=0.3, 
                     edgecolor='red', linewidth=3, label='Study Area (5km buffer)')
        
        # Sample GPS points for better visualization (max 1000 points)
        if len(gdf_plot) > 1000:
            gdf_sample = gdf_plot.sample(n=1000, random_state=42).sort_values('timestamp')
            sample_note = f" (showing {1000} of {len(gdf_plot)} points)"
        else:
            gdf_sample = gdf_plot.copy()
            sample_note = ""
        
        # Plot GPS track as connected line
        if len(gdf_sample) > 1:
            # Sort by timestamp for proper track connection
            gdf_sample = gdf_sample.sort_values('timestamp')
            
            # Plot track line
            coords = [(row[gdf_sample.geometry.name].x, row[gdf_sample.geometry.name].y) 
                     for _, row in gdf_sample.iterrows()]
            if coords:
                lons, lats = zip(*coords)
                ax.plot(lons, lats, '-', color='blue', linewidth=1.5, alpha=0.7, 
                       label='Movement track')
        
        # Plot GPS points
        gdf_sample.plot(ax=ax, color='darkblue', markersize=15, alpha=0.8,
                       edgecolors='white', linewidth=0.5, label=f'GPS fixes{sample_note}')
        
        # Mark start and end points
        if len(gdf_sample) > 1:
            start_point = gdf_sample.iloc[0]
            end_point = gdf_sample.iloc[-1]
            
            ax.scatter(start_point.geometry.x, start_point.geometry.y, 
                      c='green', s=100, marker='o', edgecolors='white', 
                      linewidth=2, label='Start', zorder=5)
            ax.scatter(end_point.geometry.x, end_point.geometry.y, 
                      c='red', s=100, marker='s', edgecolors='white', 
                      linewidth=2, label='End', zorder=5)
        
        # Add background map
        if has_contextily:
            try:
                # Use a clear, informative basemap
                ctx.add_basemap(ax, crs=gdf_plot.crs, 
                               source=ctx.providers.CartoDB.Positron,
                               alpha=0.8, zoom='auto')
            except Exception as e:
                print(f"   ⚠️  Could not add background map: {e}")
        
        # Customize the plot
        main_title = f"{study_site}"
        subtitle = f"Collar {collar_info} • {len(gdf):,} GPS points • {aoi_gdf['area_km2'].iloc[0]:,.0f} km² study area"
        
        ax.set_title(main_title, fontsize=16, fontweight='bold', pad=20)
        ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, ha='center', va='top',
               fontsize=12, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Remove axis labels for cleaner look
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=8)
        
        # Add comprehensive legend
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.95), 
                 fontsize=10, framealpha=0.9, fancybox=True)
        
        # Add detailed statistics box
        tracking_days = (gdf['timestamp'].max() - gdf['timestamp'].min()).days
        individuals = gdf['individual-local-identifier'].nunique()
        
        # Calculate some movement statistics if available
        if 'speed_kmh' in gdf.columns:
            speeds = gdf['speed_kmh'].dropna()
            if len(speeds) > 0:
                avg_speed = speeds.mean()
                max_speed = speeds.max()
                speed_text = f"• Avg speed: {avg_speed:.1f} km/h\n• Max speed: {max_speed:.1f} km/h"
            else:
                speed_text = "• Speed data: Not available"
        else:
            speed_text = "• Speed data: Not calculated"
        
        stats_text = f"""Tracking Summary:
• Duration: {tracking_days} days
• Date range: {gdf['timestamp'].min().strftime('%Y-%m-%d')} to {gdf['timestamp'].max().strftime('%Y-%m-%d')}
• Individuals: {individuals}
• GPS fixes: {len(gdf):,}
{speed_text}
• Study area: {aoi_gdf['area_km2'].iloc[0]:,.0f} km²"""
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                        alpha=0.9, edgecolor='gray'), 
               fontsize=9, fontfamily='monospace')
        
        # Add north arrow
        ax.text(0.95, 0.90, '↑ N', transform=ax.transAxes, 
               fontsize=14, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", 
                        alpha=0.9, edgecolor='black'))
        
        # Add scale reference (approximate)
        if has_contextily:
            # Add a simple scale bar
            ax.text(0.02, 0.02, "Background: © CartoDB", transform=ax.transAxes,
                   fontsize=7, alpha=0.7, style='italic')
        
        plt.tight_layout()
        
        # Save individual map
        clean_name = study_site.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        map_file = output_dir / f"{site_number:02d}_{clean_name}_{collar_info.replace(' ', '_')}.png"
        plt.savefig(map_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        
        plt.close(fig)
        
        print(f"   ✅ Saved: {map_file.name}")
        
        return {
            'map_file': map_file,
            'study_site': study_site,
            'collar_info': collar_info,
            'gps_points': len(gdf),
            'tracking_days': tracking_days,
            'area_km2': float(aoi_gdf['area_km2'].iloc[0]),
            'individuals': individuals,
            'date_range': f"{gdf['timestamp'].min().strftime('%Y-%m-%d')} to {gdf['timestamp'].max().strftime('%Y-%m-%d')}"
        }
        
    except Exception as e:
        print(f"   ❌ Failed to create map: {e}")
        return None

def create_summary_page(site_results, output_file):
    """Create a summary page with overview statistics."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Cameroon Elephant Research - Study Sites Overview', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Prepare data
    site_names = [r['study_site'] for r in site_results]
    gps_counts = [r['gps_points'] for r in site_results]
    areas = [r['area_km2'] for r in site_results]
    tracking_days = [r['tracking_days'] for r in site_results]
    
    # 1. GPS Points by Site
    bars1 = ax1.barh(range(len(site_names)), gps_counts, color='skyblue', alpha=0.8)
    ax1.set_yticks(range(len(site_names)))
    ax1.set_yticklabels([name[:20] + "..." if len(name) > 20 else name for name in site_names], fontsize=8)
    ax1.set_xlabel('GPS Points')
    ax1.set_title('GPS Points per Study Site', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars1, gps_counts)):
        ax1.text(bar.get_width() + max(gps_counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=8)
    
    # 2. Study Area Sizes
    bars2 = ax2.barh(range(len(site_names)), areas, color='lightcoral', alpha=0.8)
    ax2.set_yticks(range(len(site_names)))
    ax2.set_yticklabels([name[:20] + "..." if len(name) > 20 else name for name in site_names], fontsize=8)
    ax2.set_xlabel('Area (km²)')
    ax2.set_title('Study Area Sizes', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, area) in enumerate(zip(bars2, areas)):
        ax2.text(bar.get_width() + max(areas)*0.01, bar.get_y() + bar.get_height()/2,
                f'{area:,.0f}', va='center', fontsize=8)
    
    # 3. Tracking Duration
    bars3 = ax3.barh(range(len(site_names)), tracking_days, color='lightgreen', alpha=0.8)
    ax3.set_yticks(range(len(site_names)))
    ax3.set_yticklabels([name[:20] + "..." if len(name) > 20 else name for name in site_names], fontsize=8)
    ax3.set_xlabel('Tracking Days')
    ax3.set_title('Tracking Duration', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, days) in enumerate(zip(bars3, tracking_days)):
        ax3.text(bar.get_width() + max(tracking_days)*0.01, bar.get_y() + bar.get_height()/2,
                f'{days}', va='center', fontsize=8)
    
    # 4. Summary Statistics Table
    ax4.axis('off')
    
    # Create summary statistics
    total_points = sum(gps_counts)
    total_area = sum(areas)
    total_sites = len(site_results)
    avg_tracking = np.mean(tracking_days)
    
    summary_text = f"""
RESEARCH SUMMARY

Study Sites: {total_sites}
Countries: Cameroon & Nigeria

Total GPS Points: {total_points:,}
Total Study Area: {total_area:,.0f} km²
Average Tracking: {avg_tracking:.0f} days

Date Range: 2003 - 2024

Largest Site: {site_names[np.argmax(areas)]}
   ({max(areas):,.0f} km²)

Most GPS Data: {site_names[np.argmax(gps_counts)]}
   ({max(gps_counts):,} points)

Longest Tracking: {site_names[np.argmax(tracking_days)]}
   ({max(tracking_days)} days)
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ Created summary page: {output_file}")

def compile_to_pdf(image_files, summary_file, output_pdf):
    """Compile all images into a single PDF."""
    
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        print(f"📄 Compiling {len(image_files) + 1} pages into PDF...")
        
        with PdfPages(output_pdf) as pdf:
            # Add summary page first
            if summary_file.exists():
                img = plt.imread(summary_file)
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
                plt.close(fig)
            
            # Add individual site maps
            for img_file in sorted(image_files):
                if img_file.exists():
                    img = plt.imread(img_file)
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight', dpi=300)
                    plt.close(fig)
        
        print(f"✅ PDF created: {output_pdf}")
        return True
        
    except ImportError:
        print("❌ Cannot create PDF - matplotlib PdfPages not available")
        return False
    except Exception as e:
        print(f"❌ Failed to create PDF: {e}")
        return False

def main():
    """Main function to create individual site maps and compile PDF."""
    
    print("📚 Creating Individual Study Site Maps & PDF Compilation")
    print("=" * 70)
    
    # Setup directories
    output_dir = Path("reports/individual_site_maps")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GPS data
    data_paths = [
        Path("../GPS_Collar_CSV_Mark"),  # Relative to STEP 2 folder
        Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark"), # Absolute path - ensuring this is correctly terminated
        Path("data/raw/gps_data"), # Relative to STEP 2 folder (assuming structure from run.py)
        Path("../../GPS_Collar_CSV_Mark"), # Relative to project root if script is in STEP 2/
        # Add more potential paths here if needed
    ]
    
    data_dir = None
    for path in data_paths:
        if path.exists():
            data_dir = path
            break
    
    if not data_dir:
        print("❌ Could not find GPS data directory")
        return
    
    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        return
    
    print(f"📁 Found {len(csv_files)} GPS files")
    
    # Create individual maps
    site_results = []
    image_files = []
    
    for i, csv_file in enumerate(csv_files, 1):
        result = create_individual_site_map(csv_file, output_dir, i, len(csv_files))
        if result:
            site_results.append(result)
            image_files.append(result['map_file'])
    
    if not site_results:
        print("❌ No maps were successfully created")
        return
    
    print(f"\n📊 Creating summary overview...")
    
    # Create summary page
    summary_file = output_dir / "00_summary_overview.png"
    create_summary_page(site_results, summary_file)
    
    # Compile to PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_file = Path("reports") / f"Cameroon_Elephant_Study_Sites_{timestamp}.pdf"
    
    success = compile_to_pdf(image_files, summary_file, pdf_file)
    
    # Print final summary
    print(f"\n🎉 Individual Site Map Creation Complete!")
    print("=" * 50)
    print(f"Maps created: {len(site_results)}")
    print(f"Individual maps: {output_dir}")
    if success:
        print(f"📕 Complete PDF: {pdf_file}")
        print(f"   File size: {pdf_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"\n📋 Study Sites Processed:")
    
    # Sort by area for display
    sorted_results = sorted(site_results, key=lambda x: x['area_km2'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"  {i:2d}. {result['study_site']}")
        print(f"      {result['gps_points']:,} GPS points • {result['area_km2']:,.0f} km² • {result['tracking_days']} days")

if __name__ == "__main__":
    # Set matplotlib backend and style
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for PDF generation
    plt.style.use('default')
    
    main()