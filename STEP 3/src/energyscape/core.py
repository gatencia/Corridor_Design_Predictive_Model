#!/usr/bin/env python3
"""
Core EnergyScape energy landscape calculation engine.
Implements Pontzer's unified locomotion equations for elephant movement costs.
"""

import numpy as np
import rasterio
import rasterio.mask
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
import json
import sys

# Add paths for imports - fix the import structure
current_dir = Path(__file__).parent
src_dir = current_dir.parent
step3_root = src_dir.parent
project_root = step3_root.parent

# Add Step 1 and Step 2 to path for cross-step imports
step1_dir = project_root / "STEP 1"
step2_dir = project_root / "STEP 2"

sys.path.extend([str(src_dir), str(step1_dir), str(step2_dir / "src")])

logger = logging.getLogger(__name__)

# Import EnergyScape components with proper error handling
try:
    from pontzer_equations import PontzerEquations
    from slope_calculation import SlopeCalculator  
    from r_integration import EnerscapeRBridge, PythonEnergyCalculator, get_preferred_energy_calculator
except ImportError:
    try:
        # Try relative imports
        from .pontzer_equations import PontzerEquations
        from .slope_calculation import SlopeCalculator
        from .r_integration import EnerscapeRBridge, PythonEnergyCalculator, get_preferred_energy_calculator
    except ImportError as e:
        logger.error(f"Could not import EnergyScape core components: {e}")
        # Define fallbacks
        PontzerEquations = None
        SlopeCalculator = None

try:
    from dem_loader import DEMLoader
except ImportError:
    try:
        from ..dem_processing.dem_loader import DEMLoader
    except ImportError:
        logger.error("Could not import DEMLoader")
        DEMLoader = None

# Try to import configuration
try:
    from config.project_config import get_config
except ImportError:
    try:
        from config.energyscape_config import get_energyscape_config, get_energyscape_manager
        get_config = get_energyscape_config
    except ImportError:
        logger.warning("Could not import configuration - using defaults")
        def get_config():
            return None

class EnergyScapeProcessor:
    """
    Main class for energy landscape calculations.
    
    Orchestrates the complete workflow from DEM processing to energy surface generation,
    integrating with Step 2 AOI outputs and preparing results for Step 4 corridor analysis.
    """
    
    def __init__(self, config=None, config_file: str = None):
        """
        Initialize EnergyScape processor with configuration.
        
        Parameters:
        -----------
        config : EnergyScapeConfig, optional
            Pre-configured EnergyScape configuration object
        config_file : str, optional
            Path to YAML configuration file
        """
        # Load configuration with fallbacks
        if config is not None:
            self.config = config
        else:
            try:
                self.config = get_config()
            except:
                self.config = None
                logger.warning("Using default configuration")
        
        # Initialize components with fallbacks
        if DEMLoader:
            self.dem_loader = DEMLoader(self.config)
        else:
            self.dem_loader = None
            logger.warning("DEMLoader not available")
        
        if PontzerEquations:
            self.pontzer = PontzerEquations(
                alpha=getattr(self.config, 'pontzer_alpha', 1.0),
                beta=getattr(self.config, 'pontzer_beta', 0.75),
                gamma=getattr(self.config, 'pontzer_gamma', 1.5)
            ) if self.config else PontzerEquations()
        else:
            self.pontzer = None
            logger.error("PontzerEquations not available")
        
        if SlopeCalculator:
            self.slope_calculator = SlopeCalculator(
                resolution_m=getattr(self.config, 'dem_resolution_m', 30.0)
            ) if self.config else SlopeCalculator()
        else:
            self.slope_calculator = None
            logger.error("SlopeCalculator not available")
        
        # Get preferred energy calculator (R or Python)
        try:
            self.energy_calculator = get_preferred_energy_calculator(self.config)
        except:
            self.energy_calculator = None
            logger.warning("No energy calculator available")
        
        # Processing results tracking
        self.processing_results = {}
        
        logger.info("EnergyScape processor initialized")

    def find_step2_aoi_outputs(self) -> List[Dict[str, Any]]:
        """
        Find Step 2 AOI outputs in the actual project structure.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of discovered AOI files with metadata
        """
        logger.info("Searching for Step 2 AOI outputs...")
        
        # Define possible Step 2 output directories based on actual structure
        current_dir = Path(__file__).parent.parent.parent  # Go up to STEP 3 root
        project_root = current_dir.parent  # Go up to project root
        
        step2_output_paths = [
            project_root / "STEP 2" / "data" / "outputs",
            project_root / "STEP 2" / "outputs", 
            project_root / "STEP 2" / "data" / "processed",
            current_dir.parent / "STEP 2" / "data" / "outputs"  # Alternative path
        ]
        
        aoi_files = []
        
        for step2_path in step2_output_paths:
            if not step2_path.exists():
                logger.debug(f"Step 2 path does not exist: {step2_path}")
                continue
                
            logger.info(f"Searching in: {step2_path}")
            
            # Look for AOI files (GeoJSON and Shapefile) - including in subdirectories
            patterns = ["*aoi*.geojson", "*aoi*.shp", "*AOI*.geojson", "*AOI*.shp"]
            
            for pattern in patterns:
                found_files = list(step2_path.rglob(pattern))
                
                for aoi_file in found_files:
                    try:
                        # Load AOI to get metadata
                        logger.info(f"Found potential AOI file: {aoi_file}")
                        aoi_gdf = gpd.read_file(aoi_file)
                        
                        if len(aoi_gdf) > 0:
                            first_row = aoi_gdf.iloc[0]
                            
                            aoi_info = {
                                'path': aoi_file,
                                'format': aoi_file.suffix.lower(),
                                'crs': str(aoi_gdf.crs),
                                'area_km2': first_row.get('area_km2', 0) if 'area_km2' in first_row else 0,
                                'utm_crs': first_row.get('utm_crs', None) if 'utm_crs' in first_row else None,
                                'bounds': aoi_gdf.total_bounds.tolist(),
                                'study_site': first_row.get('study_site', aoi_file.stem) if 'study_site' in first_row else aoi_file.stem,
                                'geometry': aoi_gdf  # Store the actual geometry for processing
                            }
                            
                            aoi_files.append(aoi_info)
                            logger.info(f"✅ Valid AOI: {aoi_file.name} - {aoi_info['study_site']} ({aoi_info['area_km2']:.1f} km²)")
                        
                    except Exception as e:
                        logger.warning(f"Could not read AOI file {aoi_file}: {e}")
        
        logger.info(f"Found {len(aoi_files)} valid AOI files from Step 2")
        
        if len(aoi_files) == 0:
            logger.warning("No AOI files found! Checking if Step 2 was completed...")
            
            # Also check for any geospatial files that might be AOIs
            for step2_path in step2_output_paths:
                if step2_path.exists():
                    all_geo_files = []
                    all_geo_files.extend(step2_path.rglob("*.geojson"))
                    all_geo_files.extend(step2_path.rglob("*.shp"))
                    
                    logger.info(f"Found {len(all_geo_files)} geospatial files in {step2_path}")
                    for geo_file in all_geo_files[:5]:  # Show first 5
                        logger.info(f"  - {geo_file.name}")
        
        return aoi_files
    
    def find_suitable_dem(self, aoi_bounds: Tuple[float, float, float, float],
                         dem_search_dirs: List[Union[str, Path]] = None) -> Optional[Path]:
        """
        Find suitable DEM file for given AOI bounds.
        
        Parameters:
        -----------
        aoi_bounds : Tuple[float, float, float, float]
            AOI bounds (minx, miny, maxx, maxy)
        dem_search_dirs : List[str or Path], optional
            Directories to search for DEM files
            
        Returns:
        --------
        Optional[Path]
            Path to suitable DEM file, or None if not found
        """
        if dem_search_dirs is None:
            # Default DEM search locations using relative paths
            step3_root = Path(__file__).parent.parent.parent  # Go up to STEP 3 root
            project_root = step3_root.parent  # Go up to project root
            dem_search_dirs = [
                project_root / "STEP 2.5" / "outputs" / "robust_dems",
                project_root / "STEP 2.5" / "outputs" / "aoi_dems",
                step3_root / "data" / "raw" / "dem",
                Path("/tmp/dem"),  # Common download location
                Path.home() / "Downloads"  # User downloads
            ]
        
        if not self.dem_loader:
            logger.error("DEMLoader not available")
            return None
        
        # Find all DEM files
        dem_files = []
        for search_dir in dem_search_dirs:
            if Path(search_dir).exists():
                found_files = self.dem_loader.find_dem_files(search_dir)
                dem_files.extend(found_files)
        
        if not dem_files:
            logger.warning(f"No DEM files found in search directories: {dem_search_dirs}")
            return None
        
        logger.info(f"Found {len(dem_files)} DEM files to check")
        
        # Check which DEMs cover the AOI
        minx, miny, maxx, maxy = aoi_bounds
        
        for dem_file in dem_files:
            try:
                summary = self.dem_loader.get_dem_summary(dem_file)
                if 'bounds' in summary:
                    dem_bounds = summary['bounds']
                    
                    # Check if DEM covers AOI (with some tolerance)
                    buffer = 0.01  # ~1km buffer in degrees
                    if (dem_bounds[0] <= minx + buffer and dem_bounds[1] <= miny + buffer and
                        dem_bounds[2] >= maxx - buffer and dem_bounds[3] >= maxy - buffer):
                        
                        logger.info(f"Found suitable DEM: {dem_file.name}")
                        return dem_file
                        
            except Exception as e:
                logger.warning(f"Error checking DEM {dem_file}: {e}")
        
        logger.warning(f"No suitable DEM found for AOI bounds: {aoi_bounds}")
        return None
    
    def calculate_energy_surface_simple(self, dem_path: Union[str, Path],
                                      aoi_geometry: gpd.GeoDataFrame,
                                      body_mass_kg: float,
                                      output_path: Union[str, Path]) -> bool:
        """
        Calculate energy surface using simplified Python implementation.
        
        Parameters:
        -----------
        dem_path : str or Path
            Path to DEM file
        aoi_geometry : gpd.GeoDataFrame
            Area of Interest geometry
        body_mass_kg : float
            Elephant body mass in kilograms
        output_path : str or Path
            Path for output energy surface
            
        Returns:
        --------
        bool
            True if calculation successful
        """
        logger.info(f"Calculating energy surface for {body_mass_kg} kg elephant using Python implementation")
        
        if not self.dem_loader or not self.pontzer or not self.slope_calculator:
            logger.error("Required components not available")
            return False
        
        try:
            # Load DEM clipped to AOI
            logger.info("Loading DEM clipped to AOI...")
            dem_array, dem_profile, dem_info = self.dem_loader.load_dem_for_aoi(
                dem_path, aoi_geometry, buffer_m=1000.0
            )
            
            # Calculate slope
            logger.info("Calculating slopes...")
            slope_array = self.slope_calculator.calculate_slope_degrees(
                dem_array, dem_profile['transform']
            )
            
            # Calculate energy costs
            logger.info("Calculating energy costs...")
            energy_array = self.pontzer.calculate_cost_surface(
                mass_kg=body_mass_kg,
                slope_array=slope_array,
                nodata_value=dem_profile.get('nodata', -9999.0)
            )
            
            # Save energy surface
            logger.info(f"Saving energy surface to {output_path}")
            output_profile = dem_profile.copy()
            output_profile.update({
                'dtype': 'float32',
                'nodata': dem_profile.get('nodata', -9999.0)
            })
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                if isinstance(energy_array, np.ma.MaskedArray):
                    dst.write(energy_array.filled(output_profile['nodata']), 1)
                else:
                    dst.write(energy_array.astype(np.float32), 1)
            
            logger.info(f"✅ Energy surface calculation completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error calculating energy surface: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_all_aois(self, dem_search_dirs: List[Union[str, Path]] = None,
                        output_dir: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Process all AOIs from Step 2 to generate energy surfaces.
        
        Parameters:
        -----------
        dem_search_dirs : List[str or Path], optional
            Directories to search for DEM files
        output_dir : str or Path, optional
            Output directory for energy surfaces
            
        Returns:
        --------
        Dict[str, Any]
            Processing results summary
        """
        logger.info("🏔️ Processing all AOIs for energy landscape generation")
        
        # Setup output directory
        if output_dir is None:
            step3_root = Path(__file__).parent.parent.parent  # Go up to STEP 3 root
            output_dir = step3_root / "data" / "processed" / "energy_surfaces"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get AOIs from Step 2
        aoi_files = self.find_step2_aoi_outputs()
        
        if not aoi_files:
            logger.error("❌ No AOI files found from Step 2")
            return {'success': False, 'error': 'No AOI files found'}
        
        # Get elephant masses
        masses = {'female': 2744.0, 'male': 6029.0, 'average': 4000.0}
        if self.config and hasattr(self.config, 'get_elephant_masses'):
            masses = self.config.get_elephant_masses()
        
        # Processing results
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_aois': len(aoi_files),
            'successful_aois': 0,
            'failed_aois': 0,
            'energy_surfaces_created': [],
            'errors': []
        }
        
        # Process each AOI
        for i, aoi_info in enumerate(aoi_files, 1):
            try:
                logger.info(f"📐 Processing AOI {i}/{len(aoi_files)}: {aoi_info['study_site']}")
                
                # Use the stored geometry
                aoi_gdf = aoi_info['geometry']
                
                # Find suitable DEM
                dem_path = self.find_suitable_dem(aoi_info['bounds'], dem_search_dirs)
                
                if not dem_path:
                    error_msg = f"No suitable DEM found for {aoi_info['study_site']}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    results['failed_aois'] += 1
                    continue
                
                # Generate energy surfaces for each elephant type
                study_site = aoi_info['study_site'].replace(' ', '_').replace('/', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                site_success = False
                for sex, mass_kg in masses.items():
                    output_filename = f"energy_cost_{sex}_{study_site}_{timestamp}.tif"
                    output_path = output_dir / output_filename
                    
                    success = self.calculate_energy_surface_simple(
                        dem_path=dem_path,
                        aoi_geometry=aoi_gdf,
                        body_mass_kg=mass_kg,
                        output_path=output_path
                    )
                    
                    if success:
                        site_success = True
                        surface_info = {
                            'path': str(output_path),
                            'study_site': aoi_info['study_site'],
                            'elephant_sex': sex,
                            'body_mass_kg': mass_kg,
                            'dem_source': str(dem_path),
                            'aoi_area_km2': aoi_info['area_km2'],
                            'created': datetime.now().isoformat()
                        }
                        results['energy_surfaces_created'].append(surface_info)
                        logger.info(f"   ✅ Created energy surface: {output_filename}")
                    else:
                        error_msg = f"Failed to create energy surface for {sex} {study_site}"
                        results['errors'].append(error_msg)
                        logger.error(f"   ❌ {error_msg}")
                
                if site_success:
                    results['successful_aois'] += 1
                else:
                    results['failed_aois'] += 1
                
            except Exception as e:
                error_msg = f"Error processing AOI {aoi_info.get('study_site', 'unknown')}: {e}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                results['failed_aois'] += 1
        
        # Save processing results
        results_file = output_dir / f"energyscape_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        logger.info(f"\n🎉 EnergyScape processing completed!")
        logger.info(f"   ✅ Successful AOIs: {results['successful_aois']}/{results['total_aois']}")
        logger.info(f"   ⚡ Energy surfaces created: {len(results['energy_surfaces_created'])}")
        logger.info(f"   📄 Results saved to: {results_file}")
        
        if results['errors']:
            logger.warning(f"   ⚠️  Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                logger.warning(f"      • {error}")
        
        self.processing_results = results
        return results
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results."""
        return self.processing_results

# Alias for the main discovery method to match the expected interface
def process_step2_aoi(processor_instance):
    """Alias for finding Step 2 AOI outputs."""
    return processor_instance.find_step2_aoi_outputs()

# Monkey patch the method to maintain interface compatibility
EnergyScapeProcessor.process_step2_aoi = lambda self: self.find_step2_aoi_outputs()

# Utility functions
def run_energyscape_workflow(config_file: str = None, 
                           dem_search_dirs: List[str] = None,
                           output_dir: str = None) -> Dict[str, Any]:
    """
    Run complete EnergyScape workflow.
    
    Parameters:
    -----------
    config_file : str, optional
        Path to configuration file
    dem_search_dirs : List[str], optional
        Directories to search for DEM files
    output_dir : str, optional
        Output directory for results
        
    Returns:
    --------
    Dict[str, Any]
        Processing results
    """
    processor = EnergyScapeProcessor(config_file=config_file)
    return processor.process_all_aois(dem_search_dirs, output_dir)

# Example usage
if __name__ == "__main__":
    # Test EnergyScape core processor
    print("⚡ Testing EnergyScape Core Processor")
    print("=" * 60)
    
    # Initialize processor
    processor = EnergyScapeProcessor()
    
    # Check Step 2 AOI discovery
    print("\n🔍 Discovering Step 2 AOIs...")
    aoi_files = processor.find_step2_aoi_outputs()
    print(f"Found {len(aoi_files)} AOI files")
    
    for aoi_info in aoi_files:
        print(f"  - {aoi_info['study_site']}: {aoi_info['area_km2']:.1f} km²")
    
    print("\n🎉 EnergyScape core processor working correctly!")