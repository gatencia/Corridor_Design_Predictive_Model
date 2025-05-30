#!/usr/bin/env python3
"""
Enhanced DEM Visualization Tool with GPS and AOI Overlays
Interactive tool to visualize downloaded DEM files with GPS collar data and AOI polygons from Step 2.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import rasterio
import rasterio.plot
import geopandas as gpd
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Tuple
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# Try to import additional visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

class EnhancedDEMVisualizer:
    """Enhanced DEM visualization tool with GPS and AOI overlay capabilities."""
    
    def __init__(self, dem_directory: str = None, step2_directory: str = None):
        """
        Initialize enhanced DEM visualizer.
        
        Parameters:
        -----------
        dem_directory : str, optional
            Directory containing DEM files. If None, uses default Step 2.5 output.
        step2_directory : str, optional
            Directory containing Step 2 outputs. If None, auto-detects.
        """
        if dem_directory is None:
            self.dem_directory = Path("outputs/aoi_specific_dems")
        else:
            self.dem_directory = Path(dem_directory)
        
        if not self.dem_directory.exists():
            raise FileNotFoundError(f"DEM directory not found: {self.dem_directory}")
        
        # Auto-detect Step 2 outputs
        if step2_directory is None:
            step2_candidates = [
                Path("../STEP 2/data/outputs"),
                Path("../../STEP 2/data/outputs"),
                Path("STEP 2/data/outputs"),
                Path("../STEP 2/outputs")
            ]
            
            for candidate in step2_candidates:
                if candidate.exists():
                    self.step2_directory = candidate
                    break
            else:
                self.step2_directory = None
        else:
            self.step2_directory = Path(step2_directory)
        
        self.dem_files = self._find_dem_files()
        self.gps_data = self._load_gps_data()
        self.aoi_polygons = self._load_aoi_polygons()
        
        print(f"🗻 Enhanced DEM Visualizer initialized")
        print(f"📁 DEM directory: {self.dem_directory}")
        print(f"📊 Found {len(self.dem_files)} DEM files")
        print(f"📍 Loaded {sum(len(gps) for gps in self.gps_data.values())} GPS points from {len(self.gps_data)} collars")
        print(f"🗺️  Loaded {len(self.aoi_polygons)} AOI polygons")
    
    def _find_dem_files(self) -> List[Path]:
        """Find all DEM files in the directory."""
        extensions = ['.tif', '.tiff', '.asc', '.bil']
        dem_files = []
        
        for ext in extensions:
            dem_files.extend(list(self.dem_directory.glob(f"*{ext}")))
            dem_files.extend(list(self.dem_directory.glob(f"*{ext.upper()}")))
        
        return sorted(dem_files)
    
    def _load_gps_data(self) -> Dict[str, gpd.GeoDataFrame]:
        """Load GPS collar data from Step 2 outputs."""
        gps_data = {}
        
        if not self.step2_directory:
            print("⚠️  Step 2 directory not found - GPS overlay unavailable")
            return gps_data
        
        # Look for individual AOI directories
        individual_aois_dir = self.step2_directory / "individual_aois"
        
        if not individual_aois_dir.exists():
            print("⚠️  Individual AOIs directory not found")
            return gps_data
        
        print(f"📂 Scanning for GPS data in: {individual_aois_dir}")
        
        for aoi_dir in individual_aois_dir.iterdir():
            if aoi_dir.is_dir():
                # Look for GeoJSON files (assuming they contain GPS points)
                geojson_files = list(aoi_dir.glob("*.geojson"))
                
                for geojson_file in geojson_files:
                    try:
                        # Load the geospatial data
                        gdf = gpd.read_file(geojson_file)
                        
                        # Ensure geographic coordinates
                        if gdf.crs and not gdf.crs.is_geographic:
                            gdf = gdf.to_crs('EPSG:4326')
                        elif gdf.crs is None:
                            gdf = gdf.set_crs('EPSG:4326')
                        
                        # Extract collar identifier from directory name
                        collar_id = aoi_dir.name
                        
                        # Check if this contains point data (GPS tracks) or polygon data (AOI boundaries)
                        if gdf.geometry.geom_type.iloc[0] in ['Point', 'MultiPoint']:
                            gps_data[collar_id] = gdf
                            print(f"   📍 Loaded {len(gdf)} GPS points for {collar_id}")
                        
                    except Exception as e:
                        print(f"   ⚠️  Could not load {geojson_file.name}: {e}")
        
        return gps_data
    
    def _load_aoi_polygons(self) -> Dict[str, gpd.GeoDataFrame]:
        """Load AOI polygon data from Step 2 outputs."""
        aoi_polygons = {}
        
        if not self.step2_directory:
            return aoi_polygons
        
        individual_aois_dir = self.step2_directory / "individual_aois"
        
        if not individual_aois_dir.exists():
            return aoi_polygons
        
        for aoi_dir in individual_aois_dir.iterdir():
            if aoi_dir.is_dir():
                geojson_files = list(aoi_dir.glob("*.geojson"))
                
                for geojson_file in geojson_files:
                    try:
                        gdf = gpd.read_file(geojson_file)
                        
                        # Ensure geographic coordinates
                        if gdf.crs and not gdf.crs.is_geographic:
                            gdf = gdf.to_crs('EPSG:4326')
                        elif gdf.crs is None:
                            gdf = gdf.set_crs('EPSG:4326')
                        
                        collar_id = aoi_dir.name
                        
                        # Check if this contains polygon data (AOI boundaries)
                        if gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
                            aoi_polygons[collar_id] = gdf
                            print(f"   🗺️  Loaded AOI polygon for {collar_id}")
                        
                    except Exception as e:
                        print(f"   ⚠️  Could not load AOI polygon from {geojson_file.name}: {e}")
        
        return aoi_polygons
    
    def load_dem(self, dem_path: Path) -> Tuple[np.ndarray, Dict]:
        """Load DEM data and metadata with geographic bounds."""
        with rasterio.open(dem_path) as src:
            elevation = src.read(1, masked=True)
            
            metadata = {
                'path': dem_path,
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,  # (west, south, east, north)
                'shape': elevation.shape,
                'resolution': (src.transform[0], abs(src.transform[4])),
                'nodata': src.nodata
            }
            
            # Calculate statistics
            if hasattr(elevation, 'compressed'):
                valid_data = elevation.compressed()
            else:
                valid_data = elevation[~np.isnan(elevation)]
            
            if len(valid_data) > 0:
                metadata.update({
                    'min_elevation': float(valid_data.min()),
                    'max_elevation': float(valid_data.max()),
                    'mean_elevation': float(valid_data.mean()),
                    'std_elevation': float(valid_data.std()),
                    'valid_pixels': len(valid_data),
                    'total_pixels': elevation.size,
                    'valid_percent': (len(valid_data) / elevation.size) * 100
                })
            else:
                metadata.update({
                    'min_elevation': 0, 'max_elevation': 0, 'mean_elevation': 0,
                    'std_elevation': 0, 'valid_pixels': 0, 'total_pixels': elevation.size,
                    'valid_percent': 0
                })
        
        return elevation, metadata
    
    def find_overlapping_data(self, dem_bounds: Tuple[float, float, float, float]) -> Dict[str, gpd.GeoDataFrame]:
        """Find GPS and AOI data that overlaps with DEM bounds."""
        west, south, east, north = dem_bounds
        
        overlapping_data = {
            'gps': {},
            'aoi': {}
        }
        
        # Check GPS data
        for collar_id, gps_gdf in self.gps_data.items():
            # Filter GPS points within DEM bounds
            mask = (
                (gps_gdf.geometry.x >= west) & (gps_gdf.geometry.x <= east) &
                (gps_gdf.geometry.y >= south) & (gps_gdf.geometry.y <= north)
            )
            
            overlapping_gps = gps_gdf[mask]
            if len(overlapping_gps) > 0:
                overlapping_data['gps'][collar_id] = overlapping_gps
        
        # Check AOI polygons
        for collar_id, aoi_gdf in self.aoi_polygons.items():
            # Check if AOI polygon intersects with DEM bounds
            from shapely.geometry import box
            dem_box = box(west, south, east, north)
            
            intersects = aoi_gdf.geometry.intersects(dem_box).any()
            if intersects:
                overlapping_data['aoi'][collar_id] = aoi_gdf
        
        return overlapping_data
    
    def create_enhanced_elevation_map(self, dem_path: Path, save_path: Optional[Path] = None) -> None:
        """Create elevation map with GPS and AOI overlays."""
        elevation, metadata = self.load_dem(dem_path)
        
        # Find overlapping GPS and AOI data
        overlapping_data = self.find_overlapping_data(metadata['bounds'])
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Plot DEM as background
        west, south, east, north = metadata['bounds']
        extent = [west, east, south, north]
        
        im = ax.imshow(elevation, cmap='terrain', extent=extent, aspect='equal', alpha=0.8)
        
        # Overlay GPS points
        colors_list = plt.cm.Set1(np.linspace(0, 1, len(overlapping_data['gps'])))
        
        for i, (collar_id, gps_gdf) in enumerate(overlapping_data['gps'].items()):
            color = colors_list[i % len(colors_list)]
            
            # Plot GPS points
            ax.scatter(gps_gdf.geometry.x, gps_gdf.geometry.y, 
                      c=[color], s=2, alpha=0.6, label=f'{collar_id} GPS')
            
            print(f"   📍 Overlaid {len(gps_gdf)} GPS points for {collar_id}")
        
        # Overlay AOI polygons
        for collar_id, aoi_gdf in overlapping_data['aoi'].items():
            aoi_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, 
                        linestyle='--', alpha=0.8, label=f'{collar_id} AOI')
            
            print(f"   🗺️  Overlaid AOI polygon for {collar_id}")
        
        # Customize plot
        ax.set_title(f"Elevation Map with GPS & AOI Overlays\n{dem_path.name}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        
        # Add colorbar for elevation
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Elevation (m)", rotation=270, labelpad=20)
        
        # Add legend for GPS/AOI data
        if overlapping_data['gps'] or overlapping_data['aoi']:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"""DEM Statistics:
Elevation: {metadata['min_elevation']:.1f} - {metadata['max_elevation']:.1f} m
Mean: {metadata['mean_elevation']:.1f} ± {metadata['std_elevation']:.1f} m
Valid data: {metadata['valid_percent']:.1f}%

Overlays:
GPS points: {sum(len(gps) for gps in overlapping_data['gps'].values())}
AOI polygons: {len(overlapping_data['aoi'])}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved enhanced elevation map: {save_path}")
        else:
            plt.show()
    
    def create_collar_specific_maps(self, output_dir: Optional[Path] = None) -> None:
        """Create individual maps for each collar that has overlapping DEM data."""
        if output_dir is None:
            output_dir = self.dem_directory.parent / "enhanced_visualizations"
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n🎨 Creating collar-specific enhanced maps...")
        print(f"📁 Output directory: {output_dir}")
        
        # Group data by collar for targeted visualizations
        collar_dem_matches = {}
        
        for dem_file in self.dem_files:
            elevation, metadata = self.load_dem(dem_file)
            overlapping_data = self.find_overlapping_data(metadata['bounds'])
            
            # Find which collars have data in this DEM
            collars_in_dem = set(overlapping_data['gps'].keys()) | set(overlapping_data['aoi'].keys())
            
            for collar_id in collars_in_dem:
                if collar_id not in collar_dem_matches:
                    collar_dem_matches[collar_id] = []
                collar_dem_matches[collar_id].append(dem_file)
        
        # Create maps for each collar
        for collar_id, dem_files in collar_dem_matches.items():
            print(f"\n📍 Creating maps for {collar_id}")
            
            for dem_file in dem_files:
                self._create_collar_focused_map(collar_id, dem_file, output_dir)
    
    def _create_collar_focused_map(self, collar_id: str, dem_path: Path, output_dir: Path) -> None:
        """Create a focused map for a specific collar."""
        elevation, metadata = self.load_dem(dem_path)
        
        # Get collar-specific data
        collar_gps = self.gps_data.get(collar_id, gpd.GeoDataFrame())
        collar_aoi = self.aoi_polygons.get(collar_id, gpd.GeoDataFrame())
        
        # Filter to DEM bounds
        west, south, east, north = metadata['bounds']
        
        if not collar_gps.empty:
            mask = (
                (collar_gps.geometry.x >= west) & (collar_gps.geometry.x <= east) &
                (collar_gps.geometry.y >= south) & (collar_gps.geometry.y <= north)
            )
            collar_gps = collar_gps[mask]
        
        # Create focused visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left panel: Full DEM with overlays
        extent = [west, east, south, north]
        im1 = ax1.imshow(elevation, cmap='terrain', extent=extent, aspect='equal', alpha=0.8)
        
        if not collar_gps.empty:
            ax1.scatter(collar_gps.geometry.x, collar_gps.geometry.y, 
                       c='red', s=3, alpha=0.7, label=f'GPS Points ({len(collar_gps)})')
        
        if not collar_aoi.empty:
            collar_aoi.plot(ax=ax1, facecolor='none', edgecolor='blue', 
                           linewidth=2, linestyle='--', alpha=0.9, label='AOI Boundary')
        
        ax1.set_title(f"{collar_id}\nElevation Map with GPS Track", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Longitude (°)")
        ax1.set_ylabel("Latitude (°)")
        ax1.grid(True, alpha=0.3)
        
        if not collar_gps.empty or not collar_aoi.empty:
            ax1.legend()
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label("Elevation (m)", rotation=270, labelpad=15)
        
        # Right panel: Elevation profile or GPS density
        if not collar_gps.empty and len(collar_gps) > 10:
            # Create GPS point density heatmap
            from scipy.stats import gaussian_kde
            
            x = collar_gps.geometry.x.values
            y = collar_gps.geometry.y.values
            
            # Create a grid for density calculation
            x_grid = np.linspace(x.min(), x.max(), 50)
            y_grid = np.linspace(y.min(), y.max(), 50)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            try:
                # Calculate point density
                positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
                values = np.vstack([x, y])
                kernel = gaussian_kde(values)
                density = np.reshape(kernel(positions).T, X_grid.shape)
                
                # Plot density
                density_extent = [x.min(), x.max(), y.min(), y.max()]
                im2 = ax2.imshow(density, extent=density_extent, origin='lower', 
                               cmap='hot', alpha=0.7, aspect='equal')
                
                # Overlay actual points
                ax2.scatter(x, y, c='white', s=1, alpha=0.5)
                
                ax2.set_title(f"GPS Point Density\n{len(collar_gps)} locations", fontsize=12)
                ax2.set_xlabel("Longitude (°)")
                ax2.set_ylabel("Latitude (°)")
                
                cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
                cbar2.set_label("Point Density", rotation=270, labelpad=15)
                
            except Exception as e:
                # Fallback: simple scatter plot
                ax2.scatter(x, y, c='red', s=2, alpha=0.6)
                ax2.set_title(f"GPS Points\n{len(collar_gps)} locations", fontsize=12)
                ax2.set_xlabel("Longitude (°)")
                ax2.set_ylabel("Latitude (°)")
                ax2.grid(True, alpha=0.3)
        else:
            # Show elevation histogram instead
            if hasattr(elevation, 'compressed'):
                valid_elevations = elevation.compressed()
            else:
                valid_elevations = elevation[~np.isnan(elevation)]
            
            if len(valid_elevations) > 0:
                ax2.hist(valid_elevations, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
                ax2.set_xlabel("Elevation (m)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Elevation Distribution")
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save collar-specific map
        safe_collar_name = collar_id.replace(' ', '_').replace('/', '_')
        safe_dem_name = dem_path.stem.replace(' ', '_')
        output_filename = f"{safe_collar_name}_{safe_dem_name}_enhanced.png"
        output_path = output_dir / output_filename
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {output_filename}")
        plt.close()
    
    def create_overview_map(self, save_path: Optional[Path] = None) -> None:
        """Create overview map showing all DEM areas with GPS coverage."""
        if not self.gps_data:
            print("⚠️  No GPS data available for overview map")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot all GPS data
        colors_list = plt.cm.Set1(np.linspace(0, 1, len(self.gps_data)))
        
        all_bounds = []
        
        for i, (collar_id, gps_gdf) in enumerate(self.gps_data.items()):
            color = colors_list[i % len(colors_list)]
            
            # Plot GPS points
            ax.scatter(gps_gdf.geometry.x, gps_gdf.geometry.y, 
                      c=[color], s=1, alpha=0.6, label=f'{collar_id}')
            
            # Track bounds
            bounds = gps_gdf.total_bounds
            all_bounds.append(bounds)
        
        # Plot DEM coverage areas
        for dem_file in self.dem_files:
            _, metadata = self.load_dem(dem_file)
            west, south, east, north = metadata['bounds']
            
            # Create rectangle for DEM coverage
            rect = Rectangle((west, south), east-west, north-south, 
                           linewidth=2, edgecolor='black', facecolor='none', 
                           alpha=0.8, linestyle='-')
            ax.add_patch(rect)
            
            # Label DEM area
            ax.text(west + (east-west)/2, north + (north-south)*0.02, 
                   dem_file.stem, ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot AOI polygons
        for collar_id, aoi_gdf in self.aoi_polygons.items():
            aoi_gdf.plot(ax=ax, facecolor='none', edgecolor='red', 
                        linewidth=1, linestyle=':', alpha=0.7)
        
        ax.set_title("Overview: GPS Collar Data & DEM Coverage Areas", 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved overview map: {save_path}")
        else:
            plt.show()
    
    def visualize_all_enhanced(self, output_dir: Optional[Path] = None) -> None:
        """Create all enhanced visualizations with GPS and AOI overlays."""
        if output_dir is None:
            output_dir = self.dem_directory.parent / "enhanced_visualizations"
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n🎨 Creating enhanced visualizations with GPS & AOI overlays...")
        print(f"📁 Output directory: {output_dir}")
        
        # Overview map
        overview_path = output_dir / "overview_gps_dem_coverage.png"
        print(f"\n🗺️  Creating overview map...")
        self.create_overview_map(overview_path)
        
        # Enhanced individual DEM maps
        print(f"\n🗻 Creating enhanced DEM maps...")
        for dem_file in self.dem_files:
            base_name = dem_file.stem
            enhanced_path = output_dir / f"{base_name}_enhanced_with_overlays.png"
            self.create_enhanced_elevation_map(dem_file, enhanced_path)
        
        # Collar-specific focused maps
        print(f"\n📍 Creating collar-specific maps...")
        self.create_collar_specific_maps(output_dir)
        
        # Summary report
        self._create_summary_report(output_dir)
        
        print(f"\n🎉 All enhanced visualizations created!")
        print(f"📂 Check output directory: {output_dir}")
    
    def _create_summary_report(self, output_dir: Path) -> None:
        """Create summary report of GPS-DEM overlaps."""
        report_path = output_dir / "gps_dem_overlap_summary.json"
        
        summary = {
            'created_at': datetime.now().isoformat(),
            'dem_files': len(self.dem_files),
            'gps_collars': len(self.gps_data),
            'aoi_polygons': len(self.aoi_polygons),
            'total_gps_points': sum(len(gps) for gps in self.gps_data.values()),
            'dem_coverage': {},
            'collar_coverage': {}
        }
        
        for dem_file in self.dem_files:
            elevation, metadata = self.load_dem(dem_file)
            overlapping_data = self.find_overlapping_data(metadata['bounds'])
            
            summary['dem_coverage'][dem_file.name] = {
                'bounds': metadata['bounds'],
                'overlapping_collars': list(overlapping_data['gps'].keys()),
                'gps_points_in_area': sum(len(gps) for gps in overlapping_data['gps'].values()),
                'aoi_polygons_in_area': len(overlapping_data['aoi'])
            }
        
        for collar_id in self.gps_data.keys():
            total_points = len(self.gps_data[collar_id])
            overlapping_dems = []
            
            for dem_file in self.dem_files:
                _, metadata = self.load_dem(dem_file)
                overlapping_data = self.find_overlapping_data(metadata['bounds'])
                if collar_id in overlapping_data['gps']:
                    overlapping_dems.append(dem_file.name)
            
            summary['collar_coverage'][collar_id] = {
                'total_gps_points': total_points,
                'overlapping_dems': overlapping_dems,
                'has_aoi_polygon': collar_id in self.aoi_polygons
            }
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary report saved: {report_path}")

def main():
    """Main enhanced visualization function."""
    parser = argparse.ArgumentParser(description="Enhanced DEM visualization with GPS and AOI overlays")
    parser.add_argument("--dem-dir", type=str, help="Directory containing DEM files")
    parser.add_argument("--step2-dir", type=str, help="Directory containing Step 2 outputs")
    parser.add_argument("--output-dir", type=str, help="Output directory for visualizations")
    parser.add_argument("--overview-only", action="store_true", help="Create overview map only")
    
    args = parser.parse_args()
    
    try:
        # Initialize enhanced visualizer
        visualizer = EnhancedDEMVisualizer(args.dem_dir, args.step2_dir)
        
        if args.overview_only:
            output_path = Path(args.output_dir) / "overview_map.png" if args.output_dir else None
            visualizer.create_overview_map(output_path)
        else:
            output_dir = Path(args.output_dir) if args.output_dir else None
            visualizer.visualize_all_enhanced(output_dir)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())