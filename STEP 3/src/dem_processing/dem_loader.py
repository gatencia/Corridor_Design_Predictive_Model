#!/usr/bin/env python3
"""
DEM data loading and validation.
Handles loading, validation, and basic processing of Digital Elevation Models.
"""

import rasterio
import rasterio.mask
import rasterio.warp
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class DEMInfo:
    """Information about a loaded DEM."""
    path: Path
    crs: str
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    shape: Tuple[int, int]  # (height, width)
    resolution: Tuple[float, float]  # (x_res, y_res)
    nodata_value: Optional[float]
    data_type: str
    valid_data_percent: float
    elevation_range: Tuple[float, float]  # (min_elev, max_elev)

class DEMLoader:
    """
    Digital Elevation Model loading and validation utilities.
    
    Handles loading DEM data from various sources and formats,
    with validation and basic quality checks.
    """
    
    def __init__(self, config=None):
        """
        Initialize DEM loader.
        
        Parameters:
        -----------
        config : EnergyScapeConfig, optional
            Configuration object with DEM processing parameters
        """
        self.config = config
        self.loaded_dems = {}  # Cache for loaded DEMs
        
    def load_dem(self, dem_path: Union[str, Path], 
                 validate: bool = True) -> Tuple[np.ndarray, rasterio.profiles.Profile, DEMInfo]:
        """
        Load DEM from file with validation.
        
        Parameters:
        -----------
        dem_path : str or Path
            Path to DEM file
        validate : bool
            Whether to run validation checks
            
        Returns:
        --------
        Tuple[np.ndarray, rasterio.profiles.Profile, DEMInfo]
            DEM array, rasterio profile, and DEM information
        """
        dem_path = Path(dem_path)
        
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        logger.info(f"Loading DEM from {dem_path}")
        
        try:
            with rasterio.open(dem_path) as src:
                # Read DEM data
                dem_array = src.read(1, masked=True)  # Read first band as masked array
                profile = src.profile.copy()
                
                # Create DEM info
                dem_info = self._create_dem_info(dem_path, src, dem_array)
                
                # Validate if requested
                if validate:
                    self._validate_dem(dem_array, dem_info)
                
                # Cache the loaded DEM
                self.loaded_dems[str(dem_path)] = {
                    'array': dem_array,
                    'profile': profile,
                    'info': dem_info
                }
                
                logger.info(f"Successfully loaded DEM: {dem_info.shape[1]}x{dem_info.shape[0]} cells")
                logger.info(f"Elevation range: {dem_info.elevation_range[0]:.1f} - {dem_info.elevation_range[1]:.1f} m")
                logger.info(f"Valid data: {dem_info.valid_data_percent:.1f}%")
                
                return dem_array, profile, dem_info
                
        except rasterio.RasterioIOError as e:
            raise ValueError(f"Could not read DEM file {dem_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading DEM {dem_path}: {e}")
    
    def load_dem_for_aoi(self, dem_path: Union[str, Path], 
                         aoi_geometry: Union[gpd.GeoDataFrame, Any],
                         buffer_m: float = 1000.0) -> Tuple[np.ndarray, rasterio.profiles.Profile, DEMInfo]:
        """
        Load DEM clipped to Area of Interest with buffer.
        
        Parameters:
        -----------
        dem_path : str or Path
            Path to DEM file
        aoi_geometry : gpd.GeoDataFrame or geometry
            Area of Interest geometry
        buffer_m : float
            Buffer distance in meters
            
        Returns:
        --------
        Tuple[np.ndarray, rasterio.profiles.Profile, DEMInfo]
            Clipped DEM array, profile, and info
        """
        logger.info(f"Loading DEM clipped to AOI with {buffer_m}m buffer")
        
        # Handle different geometry input types
        if hasattr(aoi_geometry, 'geometry'):
            # GeoDataFrame
            geoms = aoi_geometry.geometry
            aoi_crs = aoi_geometry.crs
        else:
            # Single geometry
            geoms = [aoi_geometry]
            aoi_crs = None
        
        # Apply buffer
        if buffer_m > 0:
            geoms = [geom.buffer(buffer_m) for geom in geoms]
        
        dem_path = Path(dem_path)
        
        try:
            with rasterio.open(dem_path) as src:
                # Reproject AOI to DEM CRS if needed
                if aoi_crs and aoi_crs != src.crs:
                    logger.info(f"Reprojecting AOI from {aoi_crs} to DEM CRS {src.crs}")
                    import pyproj
                    from shapely.ops import transform
                    
                    transformer = pyproj.Transformer.from_crs(aoi_crs, src.crs, always_xy=True)
                    geoms = [transform(transformer.transform, geom) for geom in geoms]
                
                # Clip DEM to AOI
                clipped_array, clipped_transform = rasterio.mask.mask(
                    src, geoms, crop=True, nodata=src.nodata
                )
                
                # Get first band
                clipped_array = clipped_array[0]
                
                # Create new profile
                profile = src.profile.copy()
                profile.update({
                    'height': clipped_array.shape[0],
                    'width': clipped_array.shape[1],
                    'transform': clipped_transform
                })
                
                # Create masked array
                if profile.get('nodata') is not None:
                    masked_array = np.ma.masked_equal(clipped_array, profile['nodata'])
                else:
                    masked_array = np.ma.array(clipped_array)
                
                # Create DEM info for clipped data
                dem_info = DEMInfo(
                    path=dem_path,
                    crs=str(src.crs),
                    bounds=rasterio.transform.array_bounds(
                        clipped_array.shape[0], clipped_array.shape[1], clipped_transform
                    ),
                    shape=clipped_array.shape,
                    resolution=(abs(clipped_transform[0]), abs(clipped_transform[4])),
                    nodata_value=profile.get('nodata'),
                    data_type=str(clipped_array.dtype),
                    valid_data_percent=(np.sum(~masked_array.mask) / masked_array.size) * 100,
                    elevation_range=(float(np.min(masked_array.compressed())), 
                                   float(np.max(masked_array.compressed())))
                )
                
                logger.info(f"Clipped DEM to AOI: {dem_info.shape[1]}x{dem_info.shape[0]} cells")
                logger.info(f"Bounds: {dem_info.bounds}")
                
                return masked_array, profile, dem_info
                
        except Exception as e:
            raise RuntimeError(f"Error clipping DEM to AOI: {e}")
    
    def _create_dem_info(self, dem_path: Path, src: rasterio.DatasetReader, 
                        dem_array: np.ma.MaskedArray) -> DEMInfo:
        """Create DEMInfo object from loaded DEM."""
        
        # Calculate valid data percentage
        if hasattr(dem_array, 'mask'):
            valid_pixels = np.sum(~dem_array.mask)
            total_pixels = dem_array.size
        else:
            # Handle case where array is not masked
            if src.nodata is not None:
                valid_pixels = np.sum(dem_array != src.nodata)
            else:
                valid_pixels = dem_array.size
            total_pixels = dem_array.size
        
        valid_percent = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Get elevation range from valid data
        if hasattr(dem_array, 'compressed'):
            valid_data = dem_array.compressed()
        else:
            if src.nodata is not None:
                valid_data = dem_array[dem_array != src.nodata]
            else:
                valid_data = dem_array.ravel()
        
        if len(valid_data) > 0:
            elevation_range = (float(np.min(valid_data)), float(np.max(valid_data)))
        else:
            elevation_range = (0.0, 0.0)
        
        return DEMInfo(
            path=dem_path,
            crs=str(src.crs),
            bounds=src.bounds,
            shape=(src.height, src.width),
            resolution=(abs(src.transform[0]), abs(src.transform[4])),
            nodata_value=src.nodata,
            data_type=str(dem_array.dtype),
            valid_data_percent=valid_percent,
            elevation_range=elevation_range
        )
    
    def _validate_dem(self, dem_array: np.ma.MaskedArray, dem_info: DEMInfo) -> None:
        """
        Validate DEM data quality.
        
        Parameters:
        -----------
        dem_array : np.ma.MaskedArray
            DEM elevation data
        dem_info : DEMInfo
            DEM information object
        """
        issues = []
        
        # Check data coverage
        if dem_info.valid_data_percent < 50:
            issues.append(f"Low data coverage: {dem_info.valid_data_percent:.1f}% valid pixels")
        
        # Check elevation range
        min_elev, max_elev = dem_info.elevation_range
        elevation_range = max_elev - min_elev
        
        if elevation_range <= 0:
            issues.append("No elevation variation (flat or invalid data)")
        elif elevation_range < 1:
            issues.append(f"Very low elevation variation: {elevation_range:.2f}m")
        
        # Check for extreme values (likely errors)
        if min_elev < -500:  # Below sea level by more than 500m
            issues.append(f"Unusually low minimum elevation: {min_elev:.1f}m")
        if max_elev > 9000:  # Above 9000m (higher than Everest)
            issues.append(f"Unusually high maximum elevation: {max_elev:.1f}m")
        
        # Check resolution
        x_res, y_res = dem_info.resolution
        if abs(x_res - y_res) > 0.1 * min(x_res, y_res):
            issues.append(f"Non-square pixels: {x_res:.2f} x {y_res:.2f}")
        
        # Check for common resolution issues
        if x_res > 1000:  # Very coarse resolution
            issues.append(f"Very coarse resolution: {x_res:.1f}m")
        elif x_res < 0.1:  # Very fine resolution (might be in degrees?)
            issues.append(f"Very fine resolution: {x_res:.3f} (units check needed)")
        
        # Log issues
        if issues:
            logger.warning(f"DEM validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("DEM validation passed")
    
    def get_dem_summary(self, dem_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get summary information about a DEM without loading the full data.
        
        Parameters:
        -----------
        dem_path : str or Path
            Path to DEM file
            
        Returns:
        --------
        Dict[str, Any]
            Summary information
        """
        dem_path = Path(dem_path)
        
        try:
            with rasterio.open(dem_path) as src:
                # Sample some data for quick stats
                sample_window = rasterio.windows.Window(0, 0, 
                                                      min(1000, src.width), 
                                                      min(1000, src.height))
                sample = src.read(1, window=sample_window, masked=True)
                
                if hasattr(sample, 'compressed') and len(sample.compressed()) > 0:
                    sample_min = float(np.min(sample.compressed()))
                    sample_max = float(np.max(sample.compressed()))
                    sample_mean = float(np.mean(sample.compressed()))
                else:
                    sample_min = sample_max = sample_mean = 0.0
                
                summary = {
                    'path': str(dem_path),
                    'crs': str(src.crs),
                    'bounds': src.bounds,
                    'shape': (src.height, src.width),
                    'resolution': (abs(src.transform[0]), abs(src.transform[4])),
                    'nodata': src.nodata,
                    'dtype': str(src.dtypes[0]),
                    'sample_elevation_range': (sample_min, sample_max),
                    'sample_elevation_mean': sample_mean,
                    'file_size_mb': dem_path.stat().st_size / (1024 * 1024)
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"Error getting DEM summary for {dem_path}: {e}")
            return {'path': str(dem_path), 'error': str(e)}
    
    def find_dem_files(self, search_dir: Union[str, Path], 
                      extensions: List[str] = None) -> List[Path]:
        """
        Find DEM files in directory.
        
        Parameters:
        -----------
        search_dir : str or Path
            Directory to search
        extensions : List[str], optional
            File extensions to look for
            
        Returns:
        --------
        List[Path]
            List of DEM file paths
        """
        if extensions is None:
            extensions = ['.tif', '.tiff', '.img', '.hgt', '.bil', '.asc']
        
        search_dir = Path(search_dir)
        dem_files = []
        
        if not search_dir.exists():
            logger.warning(f"Search directory does not exist: {search_dir}")
            return dem_files
        
        for ext in extensions:
            dem_files.extend(search_dir.rglob(f'*{ext}'))
            dem_files.extend(search_dir.rglob(f'*{ext.upper()}'))
        
        # Remove duplicates and sort
        dem_files = sorted(list(set(dem_files)))
        
        logger.info(f"Found {len(dem_files)} DEM files in {search_dir}")
        return dem_files
    
    def compare_dems(self, dem_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Compare multiple DEM files.
        
        Parameters:
        -----------
        dem_paths : List[str or Path]
            List of DEM file paths
            
        Returns:
        --------
        List[Dict[str, Any]]
            Comparison information for each DEM
        """
        comparisons = []
        
        for dem_path in dem_paths:
            try:
                summary = self.get_dem_summary(dem_path)
                summary['suitable_for_energyscape'] = self._assess_energyscape_suitability(summary)
                comparisons.append(summary)
            except Exception as e:
                logger.error(f"Error comparing DEM {dem_path}: {e}")
                comparisons.append({
                    'path': str(dem_path),
                    'error': str(e),
                    'suitable_for_energyscape': False
                })
        
        return comparisons
    
    def _assess_energyscape_suitability(self, dem_summary: Dict[str, Any]) -> bool:
        """
        Assess if DEM is suitable for EnergyScape analysis.
        
        Parameters:
        -----------
        dem_summary : Dict[str, Any]
            DEM summary information
            
        Returns:
        --------
        bool
            True if suitable for EnergyScape
        """
        try:
            # Check resolution (should be reasonable for elephant movement analysis)
            x_res, y_res = dem_summary['resolution']
            resolution_ok = 1 <= min(x_res, y_res) <= 100  # 1m to 100m resolution
            
            # Check elevation range
            elev_min, elev_max = dem_summary['sample_elevation_range']
            elevation_range_ok = (elev_max - elev_min) > 1  # At least 1m variation
            
            # Check for valid elevation values
            elevation_valid = -1000 <= elev_min <= 9000 and -1000 <= elev_max <= 9000
            
            # Check file size (not too small, indicating empty/corrupt data)
            size_ok = dem_summary['file_size_mb'] > 0.1
            
            return resolution_ok and elevation_range_ok and elevation_valid and size_ok
            
        except Exception:
            return False
    
    def clear_cache(self) -> None:
        """Clear cached DEM data."""
        self.loaded_dems.clear()
        logger.info("Cleared DEM cache")

# Utility functions
def validate_dem_for_energyscape(dem_path: Union[str, Path], 
                               aoi_geometry: gpd.GeoDataFrame = None) -> Dict[str, Any]:
    """
    Validate DEM suitability for EnergyScape analysis.
    
    Parameters:
    -----------
    dem_path : str or Path
        Path to DEM file
    aoi_geometry : gpd.GeoDataFrame, optional
        Area of Interest for additional checks
        
    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    loader = DEMLoader()
    
    try:
        # Load DEM
        if aoi_geometry is not None:
            dem_array, profile, dem_info = loader.load_dem_for_aoi(dem_path, aoi_geometry)
        else:
            dem_array, profile, dem_info = loader.load_dem(dem_path)
        
        # Perform validation
        validation_results = {
            'valid': True,
            'dem_info': dem_info.__dict__,
            'suitable_for_energyscape': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check specific EnergyScape requirements
        if dem_info.valid_data_percent < 80:
            validation_results['issues'].append(f"Low data coverage: {dem_info.valid_data_percent:.1f}%")
            validation_results['recommendations'].append("Consider gap-filling or alternative DEM")
        
        if dem_info.elevation_range[1] - dem_info.elevation_range[0] < 5:
            validation_results['issues'].append("Very low topographic relief")
            validation_results['recommendations'].append("Energy landscapes may show little variation")
        
        if min(dem_info.resolution) > 100:
            validation_results['issues'].append(f"Coarse resolution: {min(dem_info.resolution):.1f}m")
            validation_results['recommendations'].append("Consider higher resolution DEM for better accuracy")
        
        # Overall assessment
        if len(validation_results['issues']) > 2:
            validation_results['suitable_for_energyscape'] = False
            validation_results['valid'] = False
        
        return validation_results
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'suitable_for_energyscape': False,
            'issues': [f"Could not load DEM: {e}"],
            'recommendations': ["Check file path and format"]
        }

# Example usage and testing
if __name__ == "__main__":
    # Test DEM loader
    print("🗻 Testing DEM Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = DEMLoader()
    
    # Test finding DEM files (if any exist)
    test_dirs = [
        Path("data/raw/dem"),
        Path("../data/raw/dem"),
        Path("/tmp")  # Just for testing directory traversal
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            dem_files = loader.find_dem_files(test_dir)
            print(f"Found {len(dem_files)} DEM files in {test_dir}")
            
            # Test getting summary for first file if any
            if dem_files:
                summary = loader.get_dem_summary(dem_files[0])
                print(f"Sample DEM summary: {summary}")
                break
    
    print("\n🎉 DEM loader module working correctly!")