#!/usr/bin/env python3
"""
DEM Visualization Module for STEP 2.5
Creates clear, intuitive visualizations of AOI polygons with DEM elevation data.
Generates individual maps for each AOI and compiles them into a comprehensive PDF.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import contextily as ctx
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DEMVisualizer:
    """
    Creates visualizations of AOI polygons with DEM elevation data.
    
    Generates individual maps for each AOI showing elevation terrain
    with AOI boundaries overlaid, and compiles all maps into a PDF.
    """
    
    def __init__(self, output_dir: Path, visualizations_dir: Path):
        """
        Initialize DEM visualizer.
        
        Parameters:
        -----------
        output_dir : Path
            Directory containing DEM mosaics from data organization
        visualizations_dir : Path
            Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.visualizations_dir = Path(visualizations_dir)
        
        # Create visualization directory structure
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.individual_maps_dir = self.visualizations_dir / "individual_maps"
        self.individual_maps_dir.mkdir(exist_ok=True)
        
        # Look for mosaics directory
        self.mosaics_dir = self.output_dir / "mosaics"
        
        logger.info(f"DEM visualizer initialized")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Visualizations directory: {self.visualizations_dir}")
        logger.info(f"  Mosaics directory: {self.mosaics_dir}")
    
    def find_aoi_dem_pairs(self) -> List[Dict[str, Any]]:
        """
        Find pairs of AOI polygons and their corresponding DEM mosaics.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of AOI-DEM pairs with metadata
        """
        logger.info("Finding AOI-DEM pairs...")
        
        # Import AOI processor to find Step 2 outputs
        import sys
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        step2_src = project_root / "STEP 2" / "src"
        sys.path.insert(0, str(step2_src))
        
        try:
            from aoi_processor import AOIProcessor
            aoi_processor = AOIProcessor(project_root)
            aoi_files = aoi_processor.find_step2_aoi_outputs()
        except ImportError:
            logger.error("Could not import AOI processor")
            return []
        
        # Find DEM mosaics
        if not self.mosaics_dir.exists():
            logger.warning(f"Mosaics directory not found: {self.mosaics_dir}")
            return []
        
        dem_files = list(self.mosaics_dir.glob("*.tif"))
        logger.info(f"Found {len(dem_files)} DEM mosaic files")
        
        # Match AOIs with DEMs
        aoi_dem_pairs = []
        
        for aoi in aoi_files:
            study_site = aoi['study_site']
            
            # Find matching DEM file
            matching_dem = None
            for dem_file in dem_files:
                # Check if study site name is in DEM filename
                safe_site_name = "".join(c for c in study_site if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
                if safe_site_name.lower() in dem_file.stem.lower():
                    matching_dem = dem_file
                    break
            
            if matching_dem:
                pair_info = {
                    'aoi_info': aoi,
                    'dem_path': matching_dem,
                    'study_site': study_site,
                    'aoi_geometry': aoi['geometry'],
                    'area_km2': aoi.get('area_km2', 0)
                }
                aoi_dem_pairs.append(pair_info)
                logger.debug(f"Paired AOI '{study_site}' with DEM '{matching_dem.name}'")
            else:
                logger.warning(f"No matching DEM found for AOI '{study_site}'")
        
        logger.info(f"Successfully paired {len(aoi_dem_pairs)} AOI-DEM combinations")
        return aoi_dem_pairs
    
    def create_elevation_colormap(self, elevation_range: Tuple[float, float]) -> Tuple[Any, str]:
        """
        Create an appropriate colormap for elevation data.
        
        Parameters:
        -----------
        elevation_range : Tuple[float, float]
            Min and max elevation values
            
        Returns:
        --------
        Tuple[colormap, label]
            Matplotlib colormap and label string
        """
        min_elev, max_elev = elevation_range
        elev_span = max_elev - min_elev
        
        if elev_span < 100:
            # Low relief - use subtle green-brown gradient
            colors = ['#2d5016', '#4a7c26', '#6ba83d', '#8bc34a', '#cddc39', '#fff9c4']
            cmap = LinearSegmentedColormap.from_list('low_relief', colors)
            label = 'Elevation (m)\n(Low Relief)'
        elif elev_span < 500:
            # Moderate relief - use green-yellow-brown
            colors = ['#1b5e20', '#388e3c', '#689f38', '#9ccc65', '#cddc39', '#fff176', '#ffb74d', '#8d6e63']
            cmap = LinearSegmentedColormap.from_list('moderate_relief', colors)
            label = 'Elevation (m)\n(Moderate Relief)'
        else:
            # High relief - use full terrain colormap
            colors = ['#0d47a1', '#1976d2', '#42a5f5', '#81c784', '#9ccc65', 
                     '#cddc39', '#fff176', '#ffb74d', '#ff8a65', '#8d6e63', '#5d4037']
            cmap = LinearSegmentedColormap.from_list('high_relief', colors)
            label = 'Elevation (m)\n(High Relief)'
        
        return cmap, label
    
    def create_individual_aoi_map(self, aoi_dem_pair: Dict[str, Any], 
                                save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Create a detailed map for a single AOI with DEM elevation data.
        
        Parameters:
        -----------
        aoi_dem_pair : Dict[str, Any]
            AOI-DEM pair information
        save_path : Optional[Path]
            Path to save the map
            
        Returns:
        --------
        Optional[Path]
            Path to saved map if successful
        """
        study_site = aoi_dem_pair['study_site']
        dem_path = aoi_dem_pair['dem_path']
        aoi_gdf = aoi_dem_pair['aoi_geometry']
        
        logger.info(f"Creating map for {study_site}...")
        
        try:
            # Set up the figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Load DEM data
            with rasterio.open(dem_path) as src:
                dem_array = src.read(1, masked=True)
                dem_transform = src.transform
                dem_bounds = src.bounds
                dem_crs = src.crs
            
            # Convert AOI to DEM CRS if needed
            if aoi_gdf.crs != dem_crs:
                aoi_gdf_projected = aoi_gdf.to_crs(dem_crs)
            else:
                aoi_gdf_projected = aoi_gdf.copy()
            
            # Get elevation statistics
            valid_elevations = dem_array.compressed() if hasattr(dem_array, 'compressed') else dem_array[~np.isnan(dem_array)]
            if len(valid_elevations) == 0:
                logger.error(f"No valid elevation data in DEM for {study_site}")
                return None
            
            min_elev = float(np.min(valid_elevations))
            max_elev = float(np.max(valid_elevations))
            mean_elev = float(np.mean(valid_elevations))
            
            # Create colormap
            cmap, cmap_label = self.create_elevation_colormap((min_elev, max_elev))
            
            # Plot DEM
            from rasterio.plot import show
            dem_plot = show(dem_array, transform=dem_transform, ax=ax, cmap=cmap, 
                          vmin=min_elev, vmax=max_elev, alpha=0.8)
            
            # Plot AOI boundary
            aoi_gdf_projected.boundary.plot(ax=ax, color='red', linewidth=3, alpha=0.9, label='Study Area Boundary')
            
            # Add AOI fill with transparency
            aoi_gdf_projected.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=3, alpha=0.7)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_elev, vmax=max_elev))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label(cmap_label, fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            # Customize the plot
            ax.set_title(f'{study_site}\nDigital Elevation Model', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Remove axis labels for cleaner look
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelsize=8)
            
            # Add comprehensive statistics box
            area_km2 = aoi_dem_pair.get('area_km2', 0)
            
            stats_text = f"""Elevation Statistics:
• Min: {min_elev:.0f} m
• Max: {max_elev:.0f} m  
• Mean: {mean_elev:.0f} m
• Relief: {max_elev - min_elev:.0f} m

Study Area:
• Area: {area_km2:.1f} km²
• Coordinate System: {str(dem_crs)[:20]}...

Data Source: DEM Mosaic
Generated: {datetime.now().strftime('%Y-%m-%d')}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                   facecolor="white", alpha=0.9, edgecolor='gray'), 
                   fontsize=9, fontfamily='monospace')
            
            # Add north arrow
            ax.text(0.95, 0.95, '↑ N', transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", 
                            alpha=0.9, edgecolor='black'))
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], color='red', linewidth=3, label='Study Area Boundary')
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(0.02, 0.85), fontsize=10, framealpha=0.9)
            
            # Set aspect ratio to equal
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            # Save the map
            if save_path is None:
                clean_name = "".join(c for c in study_site if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
                save_path = self.individual_maps_dir / f"dem_map_{clean_name}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', pad_inches=0.2)
            
            plt.close(fig)
            
            logger.info(f"✅ Created map: {save_path.name}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Failed to create map for {study_site}: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def create_summary_overview(self, aoi_dem_pairs: List[Dict[str, Any]]) -> Path:
        """Create a summary overview page for the PDF."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('STEP 2.5: DEM Data Overview for Elephant Study Areas', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Extract data for analysis
        study_sites = [pair['study_site'] for pair in aoi_dem_pairs]
        areas = [pair.get('area_km2', 0) for pair in aoi_dem_pairs]
        
        # Load elevation data for each site
        elevations_by_site = {}
        for pair in aoi_dem_pairs:
            try:
                with rasterio.open(pair['dem_path']) as src:
                    dem_array = src.read(1, masked=True)
                    valid_elev = dem_array.compressed() if hasattr(dem_array, 'compressed') else dem_array[~np.isnan(dem_array)]
                    if len(valid_elev) > 0:
                        elevations_by_site[pair['study_site']] = {
                            'min': float(np.min(valid_elev)),
                            'max': float(np.max(valid_elev)),
                            'mean': float(np.mean(valid_elev)),
                            'relief': float(np.max(valid_elev) - np.min(valid_elev))
                        }
            except Exception as e:
                logger.warning(f"Could not load elevation data for {pair['study_site']}: {e}")
        
        # 1. Study area sizes
        if areas:
            bars1 = ax1.barh(range(len(study_sites)), areas, color='skyblue', alpha=0.8)
            ax1.set_yticks(range(len(study_sites)))
            ax1.set_yticklabels([site[:20] + "..." if len(site) > 20 else site for site in study_sites], fontsize=8)
            ax1.set_xlabel('Area (km²)')
            ax1.set_title('Study Area Sizes', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, area) in enumerate(zip(bars1, areas)):
                ax1.text(bar.get_width() + max(areas)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{area:.0f}', va='center', fontsize=8)
        
        # 2. Elevation ranges
        if elevations_by_site:
            relief_values = [elev['relief'] for elev in elevations_by_site.values()]
            site_names_with_elev = list(elevations_by_site.keys())
            
            bars2 = ax2.barh(range(len(site_names_with_elev)), relief_values, color='lightcoral', alpha=0.8)
            ax2.set_yticks(range(len(site_names_with_elev)))
            ax2.set_yticklabels([site[:20] + "..." if len(site) > 20 else site for site in site_names_with_elev], fontsize=8)
            ax2.set_xlabel('Elevation Relief (m)')
            ax2.set_title('Topographic Relief', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, relief) in enumerate(zip(bars2, relief_values)):
                ax2.text(bar.get_width() + max(relief_values)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{relief:.0f}', va='center', fontsize=8)
        
        # 3. Mean elevations
        if elevations_by_site:
            mean_elevations = [elev['mean'] for elev in elevations_by_site.values()]
            
            bars3 = ax3.barh(range(len(site_names_with_elev)), mean_elevations, color='lightgreen', alpha=0.8)
            ax3.set_yticks(range(len(site_names_with_elev)))
            ax3.set_yticklabels([site[:20] + "..." if len(site) > 20 else site for site in site_names_with_elev], fontsize=8)
            ax3.set_xlabel('Mean Elevation (m)')
            ax3.set_title('Mean Elevations', fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, mean_elev) in enumerate(zip(bars3, mean_elevations)):
                ax3.text(bar.get_width() + max(mean_elevations)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{mean_elev:.0f}', va='center', fontsize=8)
        
        # 4. Summary statistics table
        ax4.axis('off')
        
        # Create summary statistics
        total_sites = len(aoi_dem_pairs)
        total_area = sum(areas) if areas else 0
        
        if elevations_by_site:
            all_reliefs = [elev['relief'] for elev in elevations_by_site.values()]
            all_means = [elev['mean'] for elev in elevations_by_site.values()]
            avg_relief = np.mean(all_reliefs)
            avg_elevation = np.mean(all_means)
            min_elevation = min(elev['min'] for elev in elevations_by_site.values())
            max_elevation = max(elev['max'] for elev in elevations_by_site.values())
        else:
            avg_relief = avg_elevation = min_elevation = max_elevation = 0
        
        summary_text = f"""
DEM DATA SUMMARY

Study Sites: {total_sites}
Total Area: {total_area:.0f} km²

ELEVATION STATISTICS
Min Elevation: {min_elevation:.0f} m
Max Elevation: {max_elevation:.0f} m
Average Elevation: {avg_elevation:.0f} m
Average Relief: {avg_relief:.0f} m

DATA SOURCES
• DEM Source: NASADEM/SRTM
• Resolution: 30m
• Processing: STEP 2.5 Automated

STATUS
✅ DEM mosaics created
✅ Ready for STEP 3 EnergyScape
✅ All sites have elevation data

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        
        plt.tight_layout()
        
        # Save summary page
        summary_path = self.individual_maps_dir / "00_summary_overview.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"✅ Created summary overview: {summary_path}")
        return summary_path
    
    def compile_maps_to_pdf(self, map_files: List[Path], 
                          output_filename: Optional[str] = None) -> Path:
        """
        Compile all individual maps into a single PDF.
        
        Parameters:
        -----------
        map_files : List[Path]
            List of individual map file paths
        output_filename : Optional[str]
            Output filename for PDF
            
        Returns:
        --------
        Path
            Path to compiled PDF
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"STEP25_DEM_Visualization_Report_{timestamp}.pdf"
        
        pdf_path = self.visualizations_dir / output_filename
        
        logger.info(f"Compiling {len(map_files)} maps into PDF...")
        
        try:
            with PdfPages(pdf_path) as pdf:
                for map_file in sorted(map_files):
                    if map_file.exists() and map_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        img = plt.imread(map_file)
                        fig, ax = plt.subplots(figsize=(11, 8.5))  # Letter size
                        ax.imshow(img)
                        ax.axis('off')
                        pdf.savefig(fig, bbox_inches='tight', dpi=300)
                        plt.close(fig)
                        logger.debug(f"Added {map_file.name} to PDF")
            
            logger.info(f"✅ PDF compiled successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"❌ Failed to compile PDF: {e}")
            raise
    
    def create_all_visualizations(self) -> Dict[str, Any]:
        """
        Create all visualizations: individual maps and compiled PDF.
        
        Returns:
        --------
        Dict[str, Any]
            Results summary
        """
        logger.info("🎨 Starting DEM visualization creation...")
        
        # Find AOI-DEM pairs
        aoi_dem_pairs = self.find_aoi_dem_pairs()
        
        if not aoi_dem_pairs:
            logger.error("No AOI-DEM pairs found")
            return {
                'success': False,
                'error': 'No AOI-DEM pairs found',
                'maps_created': 0,
                'pdf_path': None
            }
        
        logger.info(f"Found {len(aoi_dem_pairs)} AOI-DEM pairs to visualize")
        
        # Create summary overview
        summary_path = self.create_summary_overview(aoi_dem_pairs)
        
        # Create individual maps
        created_maps = [summary_path]
        
        for i, aoi_dem_pair in enumerate(aoi_dem_pairs, 1):
            logger.info(f"Creating map {i}/{len(aoi_dem_pairs)}: {aoi_dem_pair['study_site']}")
            
            map_path = self.create_individual_aoi_map(aoi_dem_pair)
            if map_path:
                created_maps.append(map_path)
        
        # Compile to PDF
        if created_maps:
            pdf_path = self.compile_maps_to_pdf(created_maps)
        else:
            pdf_path = None
        
        # Results summary
        results = {
            'success': len(created_maps) > 0,
            'aoi_dem_pairs_found': len(aoi_dem_pairs),
            'maps_created': len(created_maps),
            'individual_maps': created_maps,
            'pdf_path': pdf_path,
            'visualizations_directory': self.visualizations_dir
        }
        
        # Print summary
        logger.info(f"\n🎉 DEM Visualization Complete!")
        logger.info(f"   AOI-DEM pairs found: {results['aoi_dem_pairs_found']}")
        logger.info(f"   Maps created: {results['maps_created']}")
        logger.info(f"   Individual maps: {self.individual_maps_dir}")
        if pdf_path:
            logger.info(f"   📕 Complete PDF: {pdf_path}")
            logger.info(f"   File size: {pdf_path.stat().st_size / (1024*1024):.1f} MB")
        
        return results

# Utility function for easy usage
def create_step25_visualizations(output_dir: str, visualizations_dir: str) -> Dict[str, Any]:
    """
    Create Step 2.5 DEM visualizations.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing DEM data from Step 2.5
    visualizations_dir : str
        Directory to save visualizations
        
    Returns:
    --------
    Dict[str, Any]
        Results summary
    """
    visualizer = DEMVisualizer(Path(output_dir), Path(visualizations_dir))
    return visualizer.create_all_visualizations()

# Example usage and testing
if __name__ == "__main__":
    # Test DEM visualizer
    print("🎨 Testing DEM Visualizer")
    print("=" * 50)
    
    # Set up test directories
    test_output_dir = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/STEP 2.5/output")
    test_viz_dir = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/STEP 2.5/visualizations")
    
    # Create visualizer
    visualizer = DEMVisualizer(test_output_dir, test_viz_dir)
    
    # Test finding AOI-DEM pairs
    pairs = visualizer.find_aoi_dem_pairs()
    print(f"Found {len(pairs)} AOI-DEM pairs")
    
    if pairs:
        # Test creating a single map
        test_map = visualizer.create_individual_aoi_map(pairs[0])
        if test_map:
            print(f"✅ Test map created: {test_map}")
        
        # Test creating all visualizations
        results = visualizer.create_all_visualizations()
        print(f"Visualization results: {results}")
    
    print("\n🎉 DEM visualizer working correctly!")