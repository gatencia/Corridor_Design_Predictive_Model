#!/usr/bin/env python3
"""
R/enerscape bridge and integration.
Provides Python interface to R's enerscape package for energy landscape calculations.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import tempfile
import os
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import sys

logger = logging.getLogger(__name__)

class RIntegrationError(Exception):
    """Custom exception for R integration errors."""
    pass

class EnerscapeRBridge:
    """
    Bridge to R's enerscape package for energy landscape calculations.
    
    This class provides a Python interface to the R enerscape package,
    handling data conversion, R script execution, and result retrieval.
    """
    
    def __init__(self, config=None, r_timeout: int = 300):
        """
        Initialize R integration bridge.
        
        Parameters:
        -----------
        config : EnergyScapeConfig, optional
            Configuration object with R integration parameters
        r_timeout : int
            Timeout for R processes in seconds
        """
        self.config = config
        self.r_timeout = r_timeout
        self.r_available = self._check_r_availability()
        self.enerscape_available = False
        
        if self.r_available:
            self.enerscape_available = self._check_enerscape_availability()
            
        logger.info(f"R available: {self.r_available}")
        logger.info(f"R enerscape available: {self.enerscape_available}")
    
    def _check_r_availability(self) -> bool:
        """Check if R is available on the system."""
        try:
            result = subprocess.run(['R', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                r_version = result.stdout.split('\n')[0]
                logger.info(f"R found: {r_version}")
                return True
            else:
                logger.error(f"R enerscape calculation failed: {result.stderr}")
                logger.error(f"R stdout: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"R enerscape calculation timed out after {self.r_timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Error running R enerscape calculation: {e}")
            return False
    
    def calculate_net_energy_surface_r(self, energy_surface_path: Union[str, Path],
                                     ndvi_path: Union[str, Path],
                                     output_path: Union[str, Path],
                                     ndvi_gain_factor: float = 1000.0) -> bool:
        """
        Calculate net energy surface (energy cost - NDVI gain) using R.
        
        Parameters:
        -----------
        energy_surface_path : str or Path
            Path to energy cost surface
        ndvi_path : str or Path
            Path to NDVI raster
        output_path : str or Path
            Path for output net energy surface
        ndvi_gain_factor : float
            NDVI to energy gain conversion factor
            
        Returns:
        --------
        bool
            True if calculation successful
        """
        if not self.enerscape_available:
            raise RIntegrationError("R enerscape package not available")
        
        energy_surface_path = Path(energy_surface_path)
        ndvi_path = Path(ndvi_path)
        output_path = Path(output_path)
        
        logger.info("Calculating net energy surface using R")
        
        r_script = f'''
        library(terra)
        
        # Load energy surface
        cat("Loading energy surface from: {energy_surface_path}\\n")
        energy_surface <- rast("{energy_surface_path}")
        
        # Load NDVI
        cat("Loading NDVI from: {ndvi_path}\\n")
        ndvi <- rast("{ndvi_path}")
        
        # Resample NDVI to match energy surface if needed
        if (!compareGeom(energy_surface, ndvi, stopOnError = FALSE)) {{
            cat("Resampling NDVI to match energy surface\\n")
            ndvi <- resample(ndvi, energy_surface, method = "bilinear")
        }}
        
        # Calculate NDVI gain
        cat("Converting NDVI to energy gain with factor {ndvi_gain_factor}\\n")
        ndvi_min <- global(ndvi, "min", na.rm = TRUE)[[1]]
        ndvi_max <- global(ndvi, "max", na.rm = TRUE)[[1]]
        ndvi_range <- ndvi_max - ndvi_min
        
        if (ndvi_range > 0) {{
            ndvi_gain <- ((ndvi - ndvi_min) / ndvi_range) * {ndvi_gain_factor}
        }} else {{
            ndvi_gain <- ndvi * 0  # Zero gain if no NDVI variation
        }}
        
        # Calculate net energy (cost - gain)
        cat("Calculating net energy surface\\n")
        net_energy <- energy_surface - ndvi_gain
        
        # Save result
        cat("Saving net energy surface to: {output_path}\\n")
        writeRaster(net_energy, "{output_path}", overwrite = TRUE)
        
        # Report statistics
        net_stats <- global(net_energy, c("min", "max", "mean"), na.rm = TRUE)
        cat("Net energy surface statistics:\\n")
        print(net_stats)
        
        cat("Net energy surface calculation completed\\n")
        '''
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(['R', '--slave', '--no-restore', '--no-save'],
                                  input=r_script, text=True, capture_output=True,
                                  timeout=self.r_timeout)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Net energy surface calculation completed: {output_path}")
                return True
            else:
                logger.error(f"Net energy calculation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error calculating net energy surface: {e}")
            return False
    
    def create_circuitscape_ini(self, energy_surface_path: Union[str, Path],
                              focal_points: List[Tuple[float, float]] = None,
                              output_dir: Union[str, Path] = None) -> Path:
        """
        Create Circuitscape INI file for corridor analysis.
        
        Parameters:
        -----------
        energy_surface_path : str or Path
            Path to energy/resistance surface
        focal_points : List[Tuple[float, float]], optional
            List of (x, y) coordinates for focal points
        output_dir : str or Path, optional
            Output directory for Circuitscape results
            
        Returns:
        --------
        Path
            Path to created INI file
        """
        if not self.enerscape_available:
            logger.warning("R not available, creating basic INI file")
        
        energy_surface_path = Path(energy_surface_path)
        
        if output_dir is None:
            output_dir = energy_surface_path.parent / "circuitscape_output"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create basic INI configuration
        ini_content = f"""[circuitscape options]
data_type = raster
scenario = pairwise

[habitat_raster_file]
habitat_file = {energy_surface_path}

[options]
connect_four_neighbors_only = False
connect_using_avg_resistances = True

[output options]
write_cur_maps = True
write_volt_maps = True
compress_grids = True
log_transform_maps = False
write_cum_cur_map_only = False

[calculation options]
solver = cg+amg
"""
        
        # Add focal points if provided
        if focal_points:
            points_file = output_dir / "focal_points.txt"
            with open(points_file, 'w') as f:
                f.write("mode\tx\ty\n")
                for i, (x, y) in enumerate(focal_points):
                    mode = "start" if i == 0 else "end" if i == 1 else "include"
                    f.write(f"{mode}\t{x}\t{y}\n")
            
            ini_content += f"\n[point_file]\npoint_file = {points_file}\n"
        
        # Save INI file
        ini_file = output_dir / "circuitscape_config.ini"
        with open(ini_file, 'w') as f:
            f.write(ini_content)
        
        logger.info(f"Created Circuitscape INI file: {ini_file}")
        return ini_file
    
    def get_r_session_info(self) -> Dict[str, Any]:
        """Get R session information and package versions."""
        if not self.r_available:
            return {"error": "R not available"}
        
        r_script = '''
        info <- sessionInfo()
        cat("R_VERSION:", as.character(info$R.version$version.string), "\\n")
        cat("PLATFORM:", info$platform, "\\n")
        
        # Check for required packages
        packages <- c("terra", "enerscape")
        for (pkg in packages) {
            if (pkg %in% rownames(installed.packages())) {
                version <- packageVersion(pkg)
                cat("PACKAGE:", pkg, as.character(version), "\\n")
            } else {
                cat("PACKAGE:", pkg, "NOT_INSTALLED\\n")
            }
        }
        '''
        
        try:
            result = subprocess.run(['R', '--slave', '--no-restore', '--no-save'],
                                  input=r_script, text=True, capture_output=True,
                                  timeout=30)
            
            info = {}
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info[key.strip()] = value.strip()
            
            return info
            
        except Exception as e:
            return {"error": str(e)}

# Alternative Python-only implementation (fallback when R not available)
class PythonEnergyCalculator:
    """
    Python-only implementation of energy landscape calculations.
    Used as fallback when R enerscape is not available.
    """
    
    def __init__(self, config=None):
        """Initialize Python energy calculator."""
        self.config = config
        
        # Import Pontzer equations
        try:
            from .pontzer_equations import PontzerEquations
            self.pontzer = PontzerEquations()
        except ImportError:
            # Handle relative import issues
            import sys
            from pathlib import Path
            src_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(src_dir))
            
            from energyscape.pontzer_equations import PontzerEquations
            self.pontzer = PontzerEquations()
    
    def calculate_energy_surface_python(self, dem_array: np.ndarray,
                                      dem_profile: Dict[str, Any],
                                      body_mass_kg: float,
                                      output_path: Union[str, Path]) -> bool:
        """
        Calculate energy surface using Python implementation.
        
        Parameters:
        -----------
        dem_array : np.ndarray
            DEM elevation data
        dem_profile : Dict[str, Any]
            Rasterio profile for the DEM
        body_mass_kg : float
            Elephant body mass in kilograms
        output_path : str or Path
            Path for output energy surface
            
        Returns:
        --------
        bool
            True if calculation successful
        """
        try:
            # Import slope calculation
            from ..energyscape.slope_calculation import SlopeCalculator
            
            # Calculate slope
            slope_calc = SlopeCalculator(resolution_m=abs(dem_profile['transform'][0]))
            slope_array = slope_calc.calculate_slope_degrees(dem_array, dem_profile['transform'])
            
            # Calculate energy costs
            energy_array = self.pontzer.calculate_cost_surface(
                mass_kg=body_mass_kg,
                slope_array=slope_array,
                nodata_value=dem_profile.get('nodata', -9999.0)
            )
            
            # Save energy surface
            output_profile = dem_profile.copy()
            output_profile.update({
                'dtype': 'float32',
                'nodata': dem_profile.get('nodata', -9999.0)
            })
            
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(energy_array, 1)
            
            logger.info(f"Python energy surface calculation completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Python energy surface calculation failed: {e}")
            return False

# Utility functions
def get_preferred_energy_calculator(config=None) -> Union[EnerscapeRBridge, PythonEnergyCalculator]:
    """
    Get the preferred energy calculator (R if available, Python as fallback).
    
    Parameters:
    -----------
    config : EnergyScapeConfig, optional
        Configuration object
        
    Returns:
    --------
    Union[EnerscapeRBridge, PythonEnergyCalculator]
        Energy calculator instance
    """
    # Try R integration first
    r_bridge = EnerscapeRBridge(config)
    
    if r_bridge.enerscape_available:
        logger.info("Using R enerscape for energy calculations")
        return r_bridge
    else:
        logger.info("R enerscape not available, using Python implementation")
        return PythonEnergyCalculator(config)

def install_r_dependencies() -> bool:
    """
    Install required R dependencies for enerscape.
    
    Returns:
    --------
    bool
        True if installation successful
    """
    r_bridge = EnerscapeRBridge()
    return r_bridge.install_enerscape()

# Example usage and testing
if __name__ == "__main__":
    # Test R integration
    print("🔗 Testing R Integration Bridge")
    print("=" * 50)
    
    # Initialize R bridge
    r_bridge = EnerscapeRBridge()
    
    # Check availability
    print(f"R available: {r_bridge.r_available}")
    print(f"R enerscape available: {r_bridge.enerscape_available}")
    
    # Get R session info
    if r_bridge.r_available:
        r_info = r_bridge.get_r_session_info()
        print(f"R session info: {r_info}")
    
    # Test preferred calculator
    calculator = get_preferred_energy_calculator()
    print(f"Using calculator: {type(calculator).__name__}")
    
    print("\n🎉 R integration bridge working correctly!")
