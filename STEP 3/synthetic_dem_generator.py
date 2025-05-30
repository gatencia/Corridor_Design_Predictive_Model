#!/usr/bin/env python3
"""
Synthetic DEM Generator for Step 3 Testing
Creates realistic synthetic Digital Elevation Models for AOIs from Step 2
to bypass Step 2.5 download issues and enable Step 3 (EnergyScape) testing.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime
import sys
import logging
from scipy import ndimage
from noise import pnoise2  # For realistic terrain generation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticDEMGenerator:
    """Generate realistic synthetic DEM data for testing Step 3."""
    
    def __init__(self, resolution_m: float = 30.0):
        """
        Initialize synthetic DEM generator.
        
        Parameters:
        -----------
        resolution_m : float
            Target resolution in meters (default 30m like SRTM)
        """
        self.resolution_m = resolution_m
        self.base_elevation = 300  # Base elevation for Central Africa region
        
        # Terrain parameters for Central Africa (Cameroon/Nigeria)
        self.terrain_params = {
            'base_elevation': 300,      # Average elevation in meters
            'elevation_range': 800,     # Total elevation variation
            'ridge_frequency': 0.008,   # Large-scale ridges
            'hill_frequency': 0.02,     # Medium hills
            'detail_frequency': 0.05,   # Fine details
            'noise_amplitude': 15,      # Small-scale roughness
            'river_influence': 0.3,     # Valley/drainage influence
        }
    
    def cleanup_previous_outputs(self):
        """Clean up previous synthetic DEM outputs to avoid duplicates."""
        
        print("🧹 Cleaning up previous synthetic DEM outputs...")
        
        cleanup_dirs = [
            Path("STEP 2.5/outputs/aoi_specific_synthetic_dems"),
            Path("STEP 3/data/raw/dem"),
            Path("../STEP 2.5/outputs/aoi_specific_synthetic_dems"),
            Path("../STEP 3/data/raw/dem"),
            Path("synthetic_dem_visualizations")
        ]
        
        files_removed = 0
        
        for cleanup_dir in cleanup_dirs:
            if cleanup_dir.exists():
                print(f"   🗑️  Cleaning: {cleanup_dir}")
                
                # Remove synthetic DEM files and visualizations
                patterns = [
                    "synthetic_dem_*.tif",
                    "synthetic_dem_*.png", 
                    "terrain_*.png",
                    "elevation_*.html",
                    "*synthetic*report*.json"
                ]
                
                for pattern in patterns:
                    files = list(cleanup_dir.glob(pattern))
                    for file in files:
                        try:
                            file.unlink()
                            files_removed += 1
                            logger.debug(f"Removed: {file.name}")
                        except Exception as e:
                            logger.warning(f"Could not remove {file}: {e}")
        
        print(f"   ✅ Removed {files_removed} previous files")
        
    def create_terrain_visualization(self, terrain_array: np.ndarray, 
                                   aoi_info: Dict[str, Any],
                                   output_dir: Path) -> List[Path]:
        """Create visualizations of the synthetic terrain."""
        
        study_site = aoi_info['study_site']
        clean_name = study_site.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        
        print(f"   📊 Creating terrain visualizations for {study_site}...")
        
        viz_files = []
        
        # Create visualization directory
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. 2D Elevation Map with Contours
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Elevation heatmap
        im1 = ax1.imshow(terrain_array, cmap='terrain', origin='upper')
        ax1.set_title(f'Synthetic DEM: {study_site}\nElevation (meters)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Longitude (pixels)')
        ax1.set_ylabel('Latitude (pixels)')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Elevation (m)', rotation=270, labelpad=20)
        
        # Add contour lines
        contour_levels = np.linspace(terrain_array.min(), terrain_array.max(), 10)
        contours = ax1.contour(terrain_array, levels=contour_levels, colors='black', alpha=0.3, linewidths=0.5)
        ax1.clabel(contours, inline=True, fontsize=8, fmt='%d m')
        
        # 2. Slope visualization
        dy, dx = np.gradient(terrain_array, self.resolution_m)
        slopes = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        im2 = ax2.imshow(slopes, cmap='YlOrRd', origin='upper')
        ax2.set_title(f'Calculated Slopes\n(for Energy Surface Generation)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Longitude (pixels)')
        ax2.set_ylabel('Latitude (pixels)')
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Slope (degrees)', rotation=270, labelpad=20)
        
        # Add statistics text box
        stats_text = f"""Terrain Statistics:
Min Elevation: {terrain_array.min():.0f} m
Max Elevation: {terrain_array.max():.0f} m
Mean Elevation: {terrain_array.mean():.0f} m
Elevation Range: {terrain_array.max() - terrain_array.min():.0f} m

Slope Statistics:
Min Slope: {slopes.min():.1f}°
Max Slope: {slopes.max():.1f}°
Mean Slope: {slopes.mean():.1f}°

Resolution: {self.resolution_m} m
Size: {terrain_array.shape[1]} × {terrain_array.shape[0]} pixels"""
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save 2D visualization
        viz_2d_file = viz_dir / f"terrain_2d_{clean_name}.png"
        plt.savefig(viz_2d_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        viz_files.append(viz_2d_file)
        
        # 3. 3D Terrain Visualization
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for 3D visualization (too dense otherwise)
        step = max(1, min(terrain_array.shape) // 100)  # Keep ~100x100 points max
        terrain_sub = terrain_array[::step, ::step]
        
        # Create coordinate meshes
        x = np.arange(0, terrain_sub.shape[1])
        y = np.arange(0, terrain_sub.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Create 3D surface
        surf = ax.plot_surface(X, Y, terrain_sub, cmap='terrain', 
                              alpha=0.9, linewidth=0, antialiased=True,
                              rstride=1, cstride=1)
        
        # Customize 3D plot
        ax.set_title(f'3D Terrain Visualization: {study_site}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude (relative)')
        ax.set_ylabel('Latitude (relative)')
        ax.set_zlabel('Elevation (m)')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, label='Elevation (m)')
        
        # Set viewing angle for best perspective
        ax.view_init(elev=30, azim=45)
        
        # Save 3D visualization
        viz_3d_file = viz_dir / f"terrain_3d_{clean_name}.png"
        plt.savefig(viz_3d_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        viz_files.append(viz_3d_file)
        
        # 4. Terrain Profile Cross-sections
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Horizontal cross-section (middle row)
        mid_row = terrain_array.shape[0] // 2
        profile_h = terrain_array[mid_row, :]
        distance_h = np.arange(len(profile_h)) * self.resolution_m / 1000  # km
        
        ax1.plot(distance_h, profile_h, 'b-', linewidth=2)
        ax1.fill_between(distance_h, profile_h, alpha=0.3)
        ax1.set_title(f'Horizontal Profile (West-East): {study_site}', fontweight='bold')
        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Elevation (m)')
        ax1.grid(True, alpha=0.3)
        
        # Vertical cross-section (middle column)
        mid_col = terrain_array.shape[1] // 2
        profile_v = terrain_array[:, mid_col]
        distance_v = np.arange(len(profile_v)) * self.resolution_m / 1000  # km
        
        ax2.plot(distance_v, profile_v, 'r-', linewidth=2)
        ax2.fill_between(distance_v, profile_v, alpha=0.3, color='red')
        ax2.set_title('Vertical Profile (South-North)', fontweight='bold')
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Elevation (m)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save profile visualization  
        viz_profile_file = viz_dir / f"terrain_profiles_{clean_name}.png"
        plt.savefig(viz_profile_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        viz_files.append(viz_profile_file)
        
        print(f"   ✅ Created {len(viz_files)} visualizations")
        return viz_files
    
    def find_step2_aois(self) -> List[Path]:
        """Find AOI files from Step 2 outputs."""
        
        # Possible Step 2 output locations
        possible_paths = [
            Path("../STEP 2/data/outputs"),
            Path("../STEP 2/outputs"), 
            Path("../../STEP 2/data/outputs"),
            Path("STEP 2/data/outputs"),  # If running from project root
            Path("inputs/aois")  # Manual input location
        ]
        
        aoi_files = []
        
        for search_path in possible_paths:
            if search_path.exists():
                logger.info(f"Searching for AOIs in: {search_path}")
                
                # Look for AOI files
                patterns = ["*aoi*.geojson", "*aoi*.shp", "*AOI*.geojson", "*AOI*.shp"]
                
                for pattern in patterns:
                    found_files = list(search_path.rglob(pattern))
                    aoi_files.extend(found_files)
                
                if aoi_files:
                    logger.info(f"Found {len(aoi_files)} AOI files in {search_path}")
                    break
        
        # Remove duplicates, preferring .geojson over .shp
        unique_files = {}
        for file in aoi_files:
            stem = file.stem
            if stem not in unique_files or file.suffix == '.geojson':
                unique_files[stem] = file
        
        result = list(unique_files.values())
        logger.info(f"Selected {len(result)} unique AOI files")
        
        return result
    
    def load_aoi_bounds(self, aoi_file: Path) -> Dict[str, Any]:
        """Load AOI and extract bounds and metadata."""
        
        try:
            gdf = gpd.read_file(aoi_file)
            
            # Ensure geographic coordinates
            if gdf.crs and not gdf.crs.is_geographic:
                gdf = gdf.to_crs('EPSG:4326')
            elif gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            
            # Get bounds
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            
            # Extract metadata
            first_row = gdf.iloc[0]
            study_site = first_row.get('study_site', aoi_file.stem)
            area_km2 = first_row.get('area_km2', 0)
            
            return {
                'file_path': aoi_file,
                'study_site': str(study_site),
                'bounds': bounds,
                'area_km2': area_km2,
                'geometry': gdf.iloc[0].geometry,
                'crs': gdf.crs
            }
            
        except Exception as e:
            logger.error(f"Error loading AOI {aoi_file}: {e}")
            return None
    
    def calculate_dem_grid_params(self, bounds: np.ndarray, buffer_km: float = 2.0) -> Dict[str, Any]:
        """Calculate DEM grid parameters from geographic bounds."""
        
        minx, miny, maxx, maxy = bounds
        
        # Add buffer
        buffer_deg = buffer_km / 111.32  # Approximate km to degrees
        minx -= buffer_deg
        maxx += buffer_deg
        miny -= buffer_deg
        maxy += buffer_deg
        
        # Calculate approximate UTM zone for the area center
        center_lon = (minx + maxx) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}"  # Northern hemisphere
        
        # Calculate grid dimensions
        # Approximate conversion: 1 degree ≈ 111.32 km at equator
        width_km = (maxx - minx) * 111.32
        height_km = (maxy - miny) * 111.32
        
        # Grid size in pixels
        width_pixels = int(width_km * 1000 / self.resolution_m)
        height_pixels = int(height_km * 1000 / self.resolution_m)
        
        # Create transform for the DEM
        pixel_size_deg = self.resolution_m / 111320  # meters to degrees (approximate)
        transform = from_bounds(minx, miny, maxx, maxy, width_pixels, height_pixels)
        
        return {
            'bounds': [minx, miny, maxx, maxy],
            'width_pixels': width_pixels,
            'height_pixels': height_pixels,
            'transform': transform,
            'utm_crs': utm_crs,
            'pixel_size_deg': pixel_size_deg
        }
    
    def generate_realistic_terrain(self, width: int, height: int, 
                                 bounds: List[float]) -> np.ndarray:
        """Generate realistic terrain using Perlin noise and geological patterns."""
        
        logger.info(f"Generating terrain: {width}x{height} pixels")
        
        # Initialize elevation array
        elevation = np.zeros((height, width), dtype=np.float32)
        
        # Create coordinate grids
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Base terrain with multiple noise octaves
        base_terrain = np.zeros_like(elevation)
        
        # Large-scale topography (mountain ranges, plateaus)
        for i in range(height):
            for j in range(width):
                # Multiple octaves of Perlin noise for realistic terrain
                noise_val = 0
                
                # Large features (ridges, valleys)
                noise_val += pnoise2(X[i,j] * 4, Y[i,j] * 4, octaves=4) * 0.5
                
                # Medium features (hills)  
                noise_val += pnoise2(X[i,j] * 8, Y[i,j] * 8, octaves=2) * 0.3
                
                # Fine details
                noise_val += pnoise2(X[i,j] * 16, Y[i,j] * 16, octaves=1) * 0.2
                
                base_terrain[i,j] = noise_val
        
        # Scale to elevation range
        base_terrain = (base_terrain + 1) / 2  # Normalize to 0-1
        base_terrain = self.terrain_params['base_elevation'] + \
                      base_terrain * self.terrain_params['elevation_range']
        
        # Add geological features
        
        # 1. Add some linear ridges (simulate geological structures)
        ridge_pattern = np.sin(X * 8 * np.pi) * np.exp(-Y * 2) * 40
        base_terrain += ridge_pattern
        
        # 2. Add river valleys (drainage patterns)
        # Create some meandering valleys
        valley_x = 0.3 + 0.4 * np.sin(Y * 6 * np.pi) 
        valley_influence = np.exp(-((X - valley_x) * 5) ** 2) * 60
        base_terrain -= valley_influence
        
        # Another valley system
        valley_y = 0.6 + 0.2 * np.sin(X * 4 * np.pi)
        valley_mask = np.zeros_like(base_terrain)
        for i in range(height):
            y_norm = i / height
            valley_dist = np.abs(valley_y - y_norm)
            valley_mask[i, :] = np.exp(-(valley_dist * 8) ** 2) * 50
        base_terrain -= valley_mask
        
        # 3. Add some isolated hills
        hill_centers = [(0.2, 0.3), (0.7, 0.2), (0.5, 0.8), (0.8, 0.7)]
        for hx, hy in hill_centers:
            hill_influence = np.exp(-((X - hx) * 3) ** 2 - ((Y - hy) * 3) ** 2) * 80
            base_terrain += hill_influence
        
        # 4. Add fine-scale roughness
        roughness = np.random.normal(0, self.terrain_params['noise_amplitude'], 
                                   (height, width))
        base_terrain += roughness
        
        # 5. Smooth to remove unrealistic sharp features
        base_terrain = ndimage.gaussian_filter(base_terrain, sigma=1.0)
        
        # Ensure reasonable elevation bounds
        base_terrain = np.clip(base_terrain, 50, 1500)  # Reasonable for Central Africa
        
        logger.info(f"Terrain elevation range: {base_terrain.min():.1f} - {base_terrain.max():.1f} m")
        
        return base_terrain.astype(np.float32)
    
    def create_synthetic_dem(self, aoi_info: Dict[str, Any], 
                           output_dir: Path) -> Dict[str, Any]:
        """Create synthetic DEM for a single AOI with visualizations."""
        
        study_site = aoi_info['study_site']
        logger.info(f"Creating synthetic DEM for: {study_site}")
        
        # Calculate grid parameters
        grid_params = self.calculate_dem_grid_params(aoi_info['bounds'])
        
        # Generate terrain (fix parameter order: height first, then width)
        terrain = self.generate_realistic_terrain(
            grid_params['height_pixels'],  # height first
            grid_params['width_pixels'],   # width second  
            grid_params['bounds']
        )
        
        # Create output filename
        clean_name = study_site.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        output_file = output_dir / f"synthetic_dem_{clean_name}.tif"
        
        # Create rasterio profile
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999.0,
            'width': grid_params['width_pixels'],
            'height': grid_params['height_pixels'],
            'count': 1,
            'crs': 'EPSG:4326',  # Keep in geographic for now
            'transform': grid_params['transform'],
            'compress': 'lzw',
            'tiled': True
        }
        
        # Save DEM
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(terrain, 1)
        
        # Create visualizations
        viz_files = self.create_terrain_visualization(terrain, aoi_info, output_dir)
        
        logger.info(f"Saved synthetic DEM: {output_file.name}")
        logger.info(f"  Size: {grid_params['width_pixels']}x{grid_params['height_pixels']} pixels")
        logger.info(f"  Elevation: {terrain.min():.1f} - {terrain.max():.1f} m")
        logger.info(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        logger.info(f"  Visualizations: {len(viz_files)} images created")
        
        return {
            'dem_file': output_file,
            'visualization_files': viz_files,
            'terrain_stats': {
                'min_elevation': float(terrain.min()),
                'max_elevation': float(terrain.max()),
                'mean_elevation': float(terrain.mean()),
                'std_elevation': float(terrain.std()),
                'elevation_range': float(terrain.max() - terrain.min())
            },
            'grid_info': grid_params
        }
    
    def generate_all_synthetic_dems(self) -> Dict[str, Any]:
        """Generate synthetic DEMs for all Step 2 AOIs with cleanup and visualizations."""
        
        print("🏔️  Synthetic DEM Generator for Step 3 Testing")
        print("=" * 60)
        print("Creating realistic synthetic DEMs with visualizations")
        print("Cleaning up previous outputs to avoid duplicates")
        print()
        
        # Step 1: Clean up previous outputs
        self.cleanup_previous_outputs()
        
        # Step 2: Setup output directories
        output_dir = Path("STEP 2.5/outputs/aoi_specific_synthetic_dems")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create the expected Step 3 input directory
        step3_dem_dir = Path("STEP 3/data/raw/dem")
        step3_dem_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: Find AOI files
        aoi_files = self.find_step2_aois()
        
        if not aoi_files:
            logger.error("No AOI files found from Step 2!")
            logger.error("Ensure Step 2 has been completed successfully")
            return {'success': False, 'error': 'No AOI files found'}
        
        # Step 4: Process each AOI
        results = {
            'timestamp': datetime.now().isoformat(),
            'synthetic_dems_created': [],
            'visualizations_created': [],
            'total_aois': len(aoi_files),
            'successful': 0,
            'failed': 0,
            'output_directory': str(output_dir),
            'visualization_directory': str(output_dir / "visualizations")
        }
        
        for i, aoi_file in enumerate(aoi_files, 1):
            try:
                logger.info(f"Processing AOI {i}/{len(aoi_files)}: {aoi_file.name}")
                
                # Load AOI
                aoi_info = self.load_aoi_bounds(aoi_file)
                if not aoi_info:
                    results['failed'] += 1
                    continue
                
                # Create synthetic DEM with visualizations
                dem_result = self.create_synthetic_dem(aoi_info, output_dir)
                
                # Also copy to Step 3 expected location
                step3_dem_file = step3_dem_dir / dem_result['dem_file'].name
                shutil.copy2(dem_result['dem_file'], step3_dem_file)
                
                # Record success
                dem_info = {
                    'study_site': aoi_info['study_site'],
                    'aoi_file': str(aoi_info['file_path']),
                    'dem_file': str(dem_result['dem_file']),
                    'step3_dem_file': str(step3_dem_file),
                    'bounds': aoi_info['bounds'].tolist(),
                    'area_km2': aoi_info['area_km2'],
                    'terrain_stats': dem_result['terrain_stats'],
                    'visualization_files': [str(f) for f in dem_result['visualization_files']]
                }
                
                results['synthetic_dems_created'].append(dem_info)
                results['visualizations_created'].extend([str(f) for f in dem_result['visualization_files']])
                results['successful'] += 1
                
            except Exception as e:
                logger.error(f"Failed to create DEM for {aoi_file}: {e}")
                results['failed'] += 1
                continue
        
        # Step 5: Create summary visualization
        if results['successful'] > 0:
            self.create_summary_visualization(results, output_dir)
        
        # Step 6: Save processing report
        report_file = output_dir / f"synthetic_dem_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Step 7: Print summary
        print(f"\n🎉 Synthetic DEM Generation Complete!")
        print("=" * 40)
        print(f"AOIs processed: {results['total_aois']}")
        print(f"DEMs created: {results['successful']}")
        print(f"Visualizations: {len(results['visualizations_created'])}")
        print(f"Failed: {results['failed']}")
        print(f"Output directory: {output_dir}")
        print(f"Step 3 directory: {step3_dem_dir}")
        print(f"Report saved: {report_file}")
        
        if results['successful'] > 0:
            print(f"\n📁 Created synthetic DEMs:")
            for dem_info in results['synthetic_dems_created']:
                stats = dem_info['terrain_stats']
                print(f"  • {dem_info['study_site']} → {Path(dem_info['dem_file']).name}")
                print(f"    Elevation: {stats['min_elevation']:.0f}-{stats['max_elevation']:.0f}m "
                      f"(mean: {stats['mean_elevation']:.0f}m)")
            
            print(f"\n🖼️  Terrain Visualizations:")
            viz_dir = output_dir / "visualizations"
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*.png"))
                print(f"   📂 Location: {viz_dir}")
                print(f"   🖼️  Files: {len(viz_files)} images created")
                print(f"   Types: 2D elevation maps, 3D terrain, profiles, summary")
            
            print(f"\n✅ Step 3 is now ready to test!")
            print(f"Run: cd 'STEP 3' && python run_energyscape.py")
        
        return results
    
    def create_summary_visualization(self, results: Dict[str, Any], output_dir: Path):
        """Create a summary visualization showing all generated terrains."""
        
        print("   📊 Creating summary visualization...")
        
        try:
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            successful_dems = results['synthetic_dems_created']
            n_dems = len(successful_dems)
            
            if n_dems == 0:
                return
            
            # Create grid for subplots
            cols = min(3, n_dems)
            rows = (n_dems + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_dems == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            fig.suptitle('Synthetic DEM Summary - All Study Sites', fontsize=16, fontweight='bold')
            
            for i, dem_info in enumerate(successful_dems):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Load and display the DEM
                try:
                    with rasterio.open(dem_info['dem_file']) as src:
                        terrain = src.read(1, masked=True)
                    
                    im = ax.imshow(terrain, cmap='terrain', origin='upper')
                    ax.set_title(f"{dem_info['study_site']}\n"
                               f"{dem_info['terrain_stats']['elevation_range']:.0f}m range", 
                               fontsize=10, fontweight='bold')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, shrink=0.6, label='Elevation (m)')
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading\n{dem_info['study_site']}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"Error: {dem_info['study_site']}")
            
            # Hide unused subplots
            for i in range(n_dems, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save summary
            summary_file = viz_dir / "summary_all_synthetic_dems.png" 
            plt.savefig(summary_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   ✅ Summary visualization saved: {summary_file.name}")
            
        except Exception as e:
            logger.warning(f"Could not create summary visualization: {e}")

def main():
    """Main synthetic DEM generation workflow with visualizations."""
    
    # Check for required dependencies
    missing_deps = []
    
    required_packages = [
        ('noise', 'Perlin noise for terrain generation'),
        ('matplotlib', 'Terrain visualizations'),
        ('rasterio', 'DEM file handling'),
        ('geopandas', 'AOI file processing'),
        ('scipy', 'Terrain processing')
    ]
    
    for package, description in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_deps.append((package, description))
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for package, description in missing_deps:
            print(f"   • {package} - {description}")
        print(f"\nInstall with: pip install {' '.join([pkg for pkg, _ in missing_deps])}")
        sys.exit(1)
    
    try:
        # Initialize generator
        generator = SyntheticDEMGenerator(resolution_m=30.0)
        
        # Generate all synthetic DEMs with visualizations
        results = generator.generate_all_synthetic_dems()
        
        if results.get('successful', 0) > 0:
            print("\n🚀 Next Steps:")
            print("1. Check the visualizations to see your synthetic terrains")
            print("2. The synthetic DEMs are ready for Step 3 testing")
            print("3. Run Step 3: cd 'STEP 3' && python run_energyscape.py")
            print("4. The synthetic data will work with the EnergyScape pipeline")
            
            # Show visualization locations
            viz_count = len(results.get('visualizations_created', []))
            if viz_count > 0:
                print(f"\n🖼️  {viz_count} terrain visualizations created:")
                print(f"   📂 Location: STEP 2.5/outputs/aoi_specific_synthetic_dems/visualizations/")
                print(f"   🖼️  Check the PNG files to see your synthetic terrains!")
            
            return True
        else:
            print("\n❌ No synthetic DEMs were created successfully")
            return False
            
    except Exception as e:
        logger.error(f"Synthetic DEM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)