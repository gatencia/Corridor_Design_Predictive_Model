#!/usr/bin/env python3
"""
DEM Visualization Tool
Interactive tool to visualize downloaded DEM files with elevation maps, 3D views, and statistics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import rasterio
import rasterio.plot
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# Try to import additional visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not available. 3D visualizations will use matplotlib.")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

class DEMVisualizer:
    """Interactive DEM visualization tool."""
    
    def __init__(self, dem_directory: str = None):
        """
        Initialize DEM visualizer.
        
        Parameters:
        -----------
        dem_directory : str, optional
            Directory containing DEM files. If None, uses default Step 2.5 output.
        """
        if dem_directory is None:
            # Default to Step 2.5 outputs
            self.dem_directory = Path("outputs/aoi_specific_dems")
        else:
            self.dem_directory = Path(dem_directory)
        
        if not self.dem_directory.exists():
            raise FileNotFoundError(f"DEM directory not found: {self.dem_directory}")
        
        self.dem_files = self._find_dem_files()
        
        print(f"🗻 DEM Visualizer initialized")
        print(f"📁 DEM directory: {self.dem_directory}")
        print(f"📊 Found {len(self.dem_files)} DEM files")
    
    def _find_dem_files(self) -> List[Path]:
        """Find all DEM files in the directory."""
        extensions = ['.tif', '.tiff', '.asc', '.bil']
        dem_files = []
        
        for ext in extensions:
            dem_files.extend(list(self.dem_directory.glob(f"*{ext}")))
            dem_files.extend(list(self.dem_directory.glob(f"*{ext.upper()}")))
        
        return sorted(dem_files)
    
    def load_dem(self, dem_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load DEM data and metadata.
        
        Parameters:
        -----------
        dem_path : Path
            Path to DEM file
            
        Returns:
        --------
        Tuple[np.ndarray, Dict]
            Elevation array and metadata
        """
        with rasterio.open(dem_path) as src:
            # Read elevation data
            elevation = src.read(1, masked=True)
            
            # Get metadata
            metadata = {
                'path': dem_path,
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
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
                    'min_elevation': 0,
                    'max_elevation': 0,
                    'mean_elevation': 0,
                    'std_elevation': 0,
                    'valid_pixels': 0,
                    'total_pixels': elevation.size,
                    'valid_percent': 0
                })
        
        return elevation, metadata
    
    def create_elevation_map(self, dem_path: Path, save_path: Optional[Path] = None) -> None:
        """
        Create 2D elevation map visualization.
        
        Parameters:
        -----------
        dem_path : Path
            Path to DEM file
        save_path : Path, optional
            Path to save the figure
        """
        elevation, metadata = self.load_dem(dem_path)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Main elevation map
        im1 = ax1.imshow(elevation, cmap='terrain', aspect='equal')
        ax1.set_title(f"Elevation Map\n{dem_path.name}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Longitude (pixels)")
        ax1.set_ylabel("Latitude (pixels)")
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label("Elevation (m)", rotation=270, labelpad=20)
        
        # Elevation histogram
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
            
            # Add statistics text
            stats_text = f"""Statistics:
Min: {metadata['min_elevation']:.1f} m
Max: {metadata['max_elevation']:.1f} m
Mean: {metadata['mean_elevation']:.1f} m
Std: {metadata['std_elevation']:.1f} m
Valid: {metadata['valid_percent']:.1f}%"""
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved elevation map: {save_path}")
        else:
            plt.show()
    
    def create_3d_surface(self, dem_path: Path, save_path: Optional[Path] = None) -> None:
        """
        Create 3D surface visualization.
        
        Parameters:
        -----------
        dem_path : Path
            Path to DEM file
        save_path : Path, optional
            Path to save the figure
        """
        elevation, metadata = self.load_dem(dem_path)
        
        if PLOTLY_AVAILABLE:
            self._create_plotly_3d(elevation, metadata, save_path)
        else:
            self._create_matplotlib_3d(elevation, metadata, save_path)
    
    def _create_matplotlib_3d(self, elevation: np.ndarray, metadata: Dict, save_path: Optional[Path] = None) -> None:
        """Create 3D surface using matplotlib."""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Downsample for performance if too large
        if elevation.shape[0] > 200 or elevation.shape[1] > 200:
            step = max(elevation.shape[0] // 200, elevation.shape[1] // 200)
            elevation_ds = elevation[::step, ::step]
        else:
            elevation_ds = elevation
        
        # Create coordinate grids
        y, x = np.mgrid[0:elevation_ds.shape[0], 0:elevation_ds.shape[1]]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        if hasattr(elevation_ds, 'filled'):
            surface_data = elevation_ds.filled(np.nan)
        else:
            surface_data = elevation_ds
        
        surf = ax.plot_surface(x, y, surface_data, cmap='terrain', alpha=0.8)
        
        ax.set_title(f"3D Elevation Surface\n{metadata['path'].name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_zlabel("Elevation (m)")
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=10, label="Elevation (m)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved 3D surface: {save_path}")
        else:
            plt.show()
    
    def _create_plotly_3d(self, elevation: np.ndarray, metadata: Dict, save_path: Optional[Path] = None) -> None:
        """Create interactive 3D surface using Plotly."""
        # Downsample for performance
        if elevation.shape[0] > 300 or elevation.shape[1] > 300:
            step = max(elevation.shape[0] // 300, elevation.shape[1] // 300)
            elevation_ds = elevation[::step, ::step]
        else:
            elevation_ds = elevation
        
        if hasattr(elevation_ds, 'filled'):
            surface_data = elevation_ds.filled(np.nan)
        else:
            surface_data = elevation_ds
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            z=surface_data,
            colorscale='terrain',
            colorbar=dict(title="Elevation (m)")
        )])
        
        fig.update_layout(
            title=f"3D Elevation Surface - {metadata['path'].name}",
            scene=dict(
                xaxis_title="X (pixels)",
                yaxis_title="Y (pixels)",
                zaxis_title="Elevation (m)",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=1000,
            height=700
        )
        
        if save_path:
            fig.write_html(str(save_path).replace('.png', '.html'))
            print(f"💾 Saved interactive 3D: {save_path.with_suffix('.html')}")
        else:
            fig.show()
    
    def create_comparison_plot(self, save_path: Optional[Path] = None) -> None:
        """
        Create comparison plot of all DEM files.
        
        Parameters:
        -----------
        save_path : Path, optional
            Path to save the figure
        """
        if len(self.dem_files) == 0:
            print("❌ No DEM files found for comparison")
            return
        
        n_files = len(self.dem_files)
        cols = min(3, n_files)
        rows = (n_files + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_files == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, dem_file in enumerate(self.dem_files):
            elevation, metadata = self.load_dem(dem_file)
            
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
            
            im = ax.imshow(elevation, cmap='terrain', aspect='equal')
            ax.set_title(f"{dem_file.stem}", fontsize=10, fontweight='bold')
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Elevation (m)", rotation=270, labelpad=15, fontsize=8)
            
            # Add stats
            stats_text = f"Range: {metadata['min_elevation']:.0f}-{metadata['max_elevation']:.0f}m\nMean: {metadata['mean_elevation']:.0f}m"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_files, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved comparison plot: {save_path}")
        else:
            plt.show()
    
    def print_summary(self) -> None:
        """Print summary of all DEM files."""
        print(f"\n🗻 DEM Files Summary")
        print("=" * 80)
        
        for dem_file in self.dem_files:
            try:
                elevation, metadata = self.load_dem(dem_file)
                
                print(f"\n📄 {dem_file.name}")
                print(f"   📐 Size: {metadata['shape'][1]} x {metadata['shape'][0]} pixels")
                print(f"   📏 Resolution: {metadata['resolution'][0]:.1f} x {metadata['resolution'][1]:.1f} m/pixel")
                print(f"   🗺️  CRS: {metadata['crs']}")
                print(f"   📊 Elevation: {metadata['min_elevation']:.1f} - {metadata['max_elevation']:.1f} m")
                print(f"   📈 Mean ± Std: {metadata['mean_elevation']:.1f} ± {metadata['std_elevation']:.1f} m")
                print(f"   ✅ Valid data: {metadata['valid_percent']:.1f}% ({metadata['valid_pixels']:,} pixels)")
                
                # Determine area type
                if 'Central_Africa' in dem_file.name:
                    area_type = "Central Africa Test Area"
                elif 'Cameroon' in dem_file.name:
                    area_type = "Cameroon Test Area"
                else:
                    area_type = "Unknown Area"
                
                resolution_type = "30m (SRTMGL1)" if 'SRTMGL1' in dem_file.name else "90m (SRTMGL3)"
                
                print(f"   🌍 Area: {area_type}")
                print(f"   📡 Resolution: {resolution_type}")
                
            except Exception as e:
                print(f"   ❌ Error loading {dem_file.name}: {e}")
    
    def visualize_all(self, output_dir: Optional[Path] = None) -> None:
        """
        Create all visualizations for all DEM files.
        
        Parameters:
        -----------
        output_dir : Path, optional
            Directory to save visualizations
        """
        if output_dir is None:
            output_dir = self.dem_directory.parent / "visualizations"
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n🎨 Creating visualizations...")
        print(f"📁 Output directory: {output_dir}")
        
        # Summary
        self.print_summary()
        
        # Comparison plot
        comparison_path = output_dir / "dem_comparison.png"
        print(f"\n📊 Creating comparison plot...")
        self.create_comparison_plot(comparison_path)
        
        # Individual visualizations
        for dem_file in self.dem_files:
            print(f"\n🗻 Processing {dem_file.name}...")
            
            base_name = dem_file.stem
            
            # 2D elevation map
            elevation_path = output_dir / f"{base_name}_elevation_map.png"
            self.create_elevation_map(dem_file, elevation_path)
            
            # 3D surface
            surface_path = output_dir / f"{base_name}_3d_surface.png"
            self.create_3d_surface(dem_file, surface_path)
        
        print(f"\n🎉 All visualizations created!")
        print(f"📂 Check output directory: {output_dir}")

def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize DEM files")
    parser.add_argument("--dem-dir", type=str, help="Directory containing DEM files")
    parser.add_argument("--output-dir", type=str, help="Output directory for visualizations")
    parser.add_argument("--summary-only", action="store_true", help="Print summary only")
    parser.add_argument("--comparison-only", action="store_true", help="Create comparison plot only")
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = DEMVisualizer(args.dem_dir)
        
        if args.summary_only:
            visualizer.print_summary()
        elif args.comparison_only:
            output_path = Path(args.output_dir) / "dem_comparison.png" if args.output_dir else None
            visualizer.create_comparison_plot(output_path)
        else:
            output_dir = Path(args.output_dir) if args.output_dir else None
            visualizer.visualize_all(output_dir)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())