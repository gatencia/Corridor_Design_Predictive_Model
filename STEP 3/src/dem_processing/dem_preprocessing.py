#!/usr/bin/env python3
"""
DEM clipping, projection, and void filling utilities.
Handles preprocessing of Digital Elevation Models for energy landscape analysis.
"""

import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp
import rasterio.fill
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from scipy import ndimage
import warnings

logger = logging.getLogger(__name__)

class DEMPreprocessor:
    """
    DEM preprocessing utilities for energy landscape analysis.
    
    Handles clipping, reprojection, void filling, and other preprocessing
    operations needed to prepare DEMs for slope and energy calculations.
    """
    
    def __init__(self, config=None):
        """
        Initialize DEM preprocessor.
        
        Parameters:
        -----------
        config : EnergyScapeConfig, optional
            Configuration object with preprocessing parameters
        """
        self.config = config
        
        # Default processing parameters
        self.default_resolution = getattr(config, 'dem_resolution_m', 30.0) if config else 30.0
        self.default_nodata = getattr(config, 'dem_nodata_value', -9999.0) if config else -9999.0
        self.fill_voids = getattr(config, 'dem_fill_voids', True) if config else True
        self.smooth_iterations = getattr(config, 'dem_smooth_iterations', 1) if config else 1
    
    def clip_dem_to_aoi(self, dem_path: Union[str, Path], 
                       aoi_geometry: Union[gpd.GeoDataFrame, Any],
                       output_path: Union[str, Path],
                       buffer_m: float = 1000.0) -> bool:
        """
        Clip DEM to Area of Interest with buffer.
        
        Parameters:
        -----------
        dem_path : str or Path
            Path to input DEM file
        aoi_geometry : gpd.GeoDataFrame or geometry
            Area of Interest geometry
        output_path : str or Path
            Path for clipped DEM output
        buffer_m : float
            Buffer distance in meters
            
        Returns:
        --------
        bool
            True if clipping successful
        """
        logger.info(f"Clipping DEM to AOI with {buffer_m}m buffer")
        
        try:
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
            
            with rasterio.open(dem_path) as src:
                # Reproject AOI to DEM CRS if needed
                if aoi_crs and aoi_crs != src.crs:
                    logger.info(f"Reprojecting AOI from {aoi_crs} to DEM CRS {src.crs}")
                    import pyproj
                    from shapely.ops import transform
                    
                    transformer = pyproj.Transformer.from_crs(aoi_crs, src.crs, always_xy=True)
                    geoms = [transform(transformer.transform, geom) for geom in geoms]
                
                # Clip DEM
                clipped_array, clipped_transform = rasterio.mask.mask(
                    src, geoms, crop=True, nodata=src.nodata
                )
                
                # Update profile
                profile = src.profile.copy()
                profile.update({
                    'height': clipped_array.shape[1],
                    'width': clipped_array.shape[2],
                    'transform': clipped_transform
                })
                
                # Ensure output directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Save clipped DEM
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(clipped_array)
                
                logger.info(f"DEM clipped successfully: {output_path}")
                logger.info(f"Output dimensions: {profile['height']}x{profile['width']}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error clipping DEM: {e}")
            return False
    
    def reproject_dem(self, input_path: Union[str, Path], 
                     output_path: Union[str, Path],
                     target_crs: Union[str, CRS],
                     target_resolution: float = None,
                     resampling_method: str = 'bilinear') -> bool:
        """
        Reproject DEM to target CRS and resolution.
        
        Parameters:
        -----------
        input_path : str or Path
            Path to input DEM
        output_path : str or Path
            Path for reprojected DEM
        target_crs : str or CRS
            Target coordinate reference system
        target_resolution : float, optional
            Target resolution in target CRS units
        resampling_method : str
            Resampling method ('nearest', 'bilinear', 'cubic', etc.)
            
        Returns:
        --------
        bool
            True if reprojection successful
        """
        logger.info(f"Reprojecting DEM to {target_crs}")
        
        try:
            with rasterio.open(input_path) as src:
                # Get resampling method
                resampling = getattr(rasterio.enums.Resampling, resampling_method.lower())
                
                # Calculate transform and dimensions for target CRS
                if target_resolution is None:
                    target_resolution = self.default_resolution
                
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds,
                    resolution=target_resolution
                )
                
                # Update profile
                profile = src.profile.copy()
                profile.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                # Ensure output directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Reproject and save
                with rasterio.open(output_path, 'w', **profile) as dst:
                    rasterio.warp.reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=resampling
                    )
                
                logger.info(f"DEM reprojected successfully: {output_path}")
                logger.info(f"New dimensions: {height}x{width}, resolution: {target_resolution}m")
                
                return True
                
        except Exception as e:
            logger.error(f"Error reprojecting DEM: {e}")
            return False
    
    def fill_voids(self, dem_array: np.ndarray, 
                  nodata_value: float = None,
                  max_search_distance: int = 100) -> np.ndarray:
        """
        Fill voids in DEM using interpolation.
        
        Parameters:
        -----------
        dem_array : np.ndarray
            DEM elevation data with voids
        nodata_value : float, optional
            NoData value to fill
        max_search_distance : int
            Maximum distance to search for valid values
            
        Returns:        
        --------
        np.ndarray
            DEM with voids filled
        """
        if nodata_value is None:
            nodata_value = self.default_nodata
        
        logger.info("Filling DEM voids...")
        
        # Create mask of valid data
        if isinstance(dem_array, np.ma.MaskedArray):
            mask = dem_array.mask
            filled_array = dem_array.data.copy()
        else:
            mask = (dem_array == nodata_value) | np.isnan(dem_array)
            filled_array = dem_array.copy()
        
        # Count voids
        void_count = np.sum(mask)
        total_pixels = dem_array.size
        void_percent = (void_count / total_pixels) * 100
        
        logger.info(f"Voids to fill: {void_count} ({void_percent:.2f}% of total)")
        
        if void_count == 0:
            logger.info("No voids found - returning original array")
            return dem_array
        
        # Use rasterio's fillnodata function if available
        try:
            # Create temporary masked array for rasterio
            filled_array = rasterio.fill.fillnodata(
                filled_array, 
                mask=~mask,  # rasterio expects True for valid data
                max_search_distance=max_search_distance
            )
            
            logger.info("Voids filled using rasterio fillnodata")
            
        except Exception as e:
            logger.warning(f"rasterio fillnodata failed: {e}, using scipy interpolation")
            
            # Fallback to scipy interpolation
            try:
                # Find indices of valid and invalid pixels
                valid_mask = ~mask
                
                if np.sum(valid_mask) == 0:
                    logger.error("No valid pixels found for interpolation")
                    return filled_array
                
                # Use scipy's distance transform and interpolation
                from scipy import ndimage
                
                # Create distance transform
                distances, indices = ndimage.distance_transform_edt(
                    mask, return_indices=True
                )
                
                # Fill voids with nearest valid values
                filled_array[mask] = filled_array[tuple(indices[:, mask])]
                
                logger.info("Voids filled using scipy interpolation")
                
            except Exception as e2:
                logger.error(f"Scipy interpolation also failed: {e2}")
                # Return original array if all filling methods fail
                return dem_array
        
        # Verify filling
        remaining_voids = np.sum(np.isnan(filled_array)) + np.sum(filled_array == nodata_value)
        if remaining_voids > 0:
            logger.warning(f"Still {remaining_voids} voids remaining after filling")
        else:
            logger.info("All voids successfully filled")
        
        return filled_array
    
    def smooth_dem(self, dem_array: np.ndarray, 
                  iterations: int = None,
                  kernel_size: int = 3) -> np.ndarray:
        """
        Apply smoothing to DEM to reduce noise.
        
        Parameters:
        -----------
        dem_array : np.ndarray
            DEM elevation data
        iterations : int, optional
            Number of smoothing iterations
        kernel_size : int
            Size of smoothing kernel (3, 5, etc.)
            
        Returns:
        --------
        np.ndarray
            Smoothed DEM
        """
        if iterations is None:
            iterations = self.smooth_iterations
        
        if iterations <= 0:
            return dem_array
        
        logger.info(f"Smoothing DEM with {iterations} iterations")
        
        # Handle masked arrays
        if isinstance(dem_array, np.ma.MaskedArray):
            data = dem_array.data.copy()
            mask = dem_array.mask
        else:
            data = dem_array.copy()
            mask = np.isnan(data) | (data == self.default_nodata)
        
        # Create smoothing kernel
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        # Apply smoothing iterations
        for i in range(iterations):
            # Only smooth valid pixels
            valid_data = np.where(mask, np.nan, data)
            
            # Apply convolution
            smoothed = ndimage.convolve(valid_data, kernel, mode='nearest')
            
            # Update only valid pixels
            data[~mask] = smoothed[~mask]
        
        logger.info("DEM smoothing completed")
        
        # Return in same format as input
        if isinstance(dem_array, np.ma.MaskedArray):
            return np.ma.array(data, mask=mask)
        else:
            return data
    
    def preprocess_dem_for_energyscape(self, dem_path: Union[str, Path],
                                     aoi_geometry: Union[gpd.GeoDataFrame, Any],
                                     output_path: Union[str, Path],
                                     target_crs: str = None,
                                     buffer_m: float = 1000.0) -> bool:
        """
        Complete DEM preprocessing workflow for EnergyScape analysis.
        
        Parameters:
        -----------
        dem_path : str or Path
            Path to input DEM
        aoi_geometry : gpd.GeoDataFrame or geometry
            Area of Interest geometry
        output_path : str or Path
            Path for preprocessed DEM
        target_crs : str, optional
            Target CRS (uses AOI CRS if None)
        buffer_m : float
            Buffer distance in meters
            
        Returns:
        --------
        bool
            True if preprocessing successful
        """
        logger.info("Starting complete DEM preprocessing workflow")
        
        try:
            # Determine target CRS
            if target_crs is None:
                if hasattr(aoi_geometry, 'crs'):
                    target_crs = aoi_geometry.crs
                else:
                    # Default to UTM zone for the area
                    logger.warning("No target CRS specified, using WGS84")
                    target_crs = 'EPSG:4326'
            
            # Create temporary files for intermediate steps
            temp_dir = Path(output_path).parent / "temp_preprocessing"
            temp_dir.mkdir(exist_ok=True)
            
            temp_clipped = temp_dir / "clipped.tif"
            temp_reprojected = temp_dir / "reprojected.tif"
            
            # Step 1: Clip to AOI
            logger.info("Step 1: Clipping DEM to AOI...")
            if not self.clip_dem_to_aoi(dem_path, aoi_geometry, temp_clipped, buffer_m):
                return False
            
            # Step 2: Reproject if needed
            with rasterio.open(temp_clipped) as src:
                if str(src.crs) != str(target_crs):
                    logger.info(f"Step 2: Reprojecting from {src.crs} to {target_crs}...")
                    if not self.reproject_dem(temp_clipped, temp_reprojected, target_crs):
                        return False
                    current_dem = temp_reprojected
                else:
                    logger.info("Step 2: No reprojection needed")
                    current_dem = temp_clipped
            
            # Step 3: Load for void filling and smoothing
            logger.info("Step 3: Loading DEM for processing...")
            with rasterio.open(current_dem) as src:
                dem_array = src.read(1, masked=True)
                profile = src.profile.copy()
            
            # Step 4: Fill voids if enabled
            if self.fill_voids:
                logger.info("Step 4: Filling voids...")
                dem_array = self.fill_voids(dem_array, profile.get('nodata'))
            else:
                logger.info("Step 4: Skipping void filling (disabled)")
            
            # Step 5: Smooth if enabled
            if self.smooth_iterations > 0:
                logger.info(f"Step 5: Smoothing ({self.smooth_iterations} iterations)...")
                dem_array = self.smooth_dem(dem_array, self.smooth_iterations)
            else:
                logger.info("Step 5: Skipping smoothing (disabled)")
            
            # Step 6: Save final result
            logger.info("Step 6: Saving preprocessed DEM...")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                if isinstance(dem_array, np.ma.MaskedArray):
                    dst.write(dem_array.filled(profile.get('nodata', self.default_nodata)), 1)
                else:
                    dst.write(dem_array, 1)
            
            # Clean up temporary files
            try:
                if temp_clipped.exists():
                    temp_clipped.unlink()
                if temp_reprojected.exists():
                    temp_reprojected.unlink()
                temp_dir.rmdir()
            except:
                pass  # Ignore cleanup errors
            
            logger.info(f"DEM preprocessing completed successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error in DEM preprocessing workflow: {e}")
            return False
    
    def validate_preprocessed_dem(self, dem_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate preprocessed DEM for EnergyScape analysis.
        
        Parameters:
        -----------
        dem_path : str or Path
            Path to preprocessed DEM
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        validation = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            with rasterio.open(dem_path) as src:
                dem_array = src.read(1, masked=True)
                
                # Basic statistics
                if isinstance(dem_array, np.ma.MaskedArray):
                    valid_data = dem_array.compressed()
                else:
                    valid_data = dem_array[~np.isnan(dem_array)]
                
                if len(valid_data) == 0:
                    validation['valid'] = False
                    validation['issues'].append("No valid elevation data")
                    return validation
                
                stats = {
                    'min_elevation': float(np.min(valid_data)),
                    'max_elevation': float(np.max(valid_data)),
                    'mean_elevation': float(np.mean(valid_data)),
                    'std_elevation': float(np.std(valid_data)),
                    'valid_pixels': len(valid_data),
                    'total_pixels': dem_array.size,
                    'data_coverage_percent': (len(valid_data) / dem_array.size) * 100
                }
                
                validation['statistics'] = stats
                
                # Validation checks
                if stats['data_coverage_percent'] < 80:
                    validation['warnings'].append(f"Low data coverage: {stats['data_coverage_percent']:.1f}%")
                
                if stats['max_elevation'] - stats['min_elevation'] < 1:
                    validation['warnings'].append("Very low elevation variation")
                
                if stats['min_elevation'] < -1000 or stats['max_elevation'] > 9000:
                    validation['issues'].append("Unrealistic elevation values")
                    validation['valid'] = False
                
                # Check CRS
                if not src.crs.is_projected:
                    validation['warnings'].append("DEM is not in projected coordinates")
                
                logger.info(f"DEM validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
                
        except Exception as e:
            validation['valid'] = False
            validation['issues'].append(f"Could not read DEM: {e}")
        
        return validation

# Utility functions
def preprocess_dem_simple(dem_path: str, aoi_path: str, output_path: str) -> bool:
    """
    Simple DEM preprocessing function.
    
    Parameters:
    -----------
    dem_path : str
        Path to input DEM
    aoi_path : str
        Path to AOI shapefile/geojson
    output_path : str
        Path for output DEM
        
    Returns:
    --------
    bool
        True if successful
    """
    preprocessor = DEMPreprocessor()
    aoi_gdf = gpd.read_file(aoi_path)
    
    return preprocessor.preprocess_dem_for_energyscape(
        dem_path, aoi_gdf, output_path
    )

# Example usage and testing
if __name__ == "__main__":
    # Test DEM preprocessing
    print("🔧 Testing DEM Preprocessing")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DEMPreprocessor()
    
    # Create synthetic test data
    print("Creating synthetic test DEM...")
    rows, cols = 200, 200
    x = np.linspace(0, 1000, cols)  # 1km extent
    y = np.linspace(0, 1000, rows)
    X, Y = np.meshgrid(x, y)
    
    # Synthetic elevation with some noise and voids
    synthetic_dem = 100 + 50 * np.sin(X/200) + 30 * np.cos(Y/150) + np.random.normal(0, 2, (rows, cols))
    
    # Add some voids
    void_mask = np.random.random((rows, cols)) < 0.05  # 5% voids
    synthetic_dem[void_mask] = np.nan
    
    print(f"Created {rows}x{cols} synthetic DEM with {np.sum(void_mask)} voids")
    
    # Test void filling
    print("Testing void filling...")
    filled_dem = preprocessor.fill_voids(synthetic_dem)
    remaining_voids = np.sum(np.isnan(filled_dem))
    print(f"Voids after filling: {remaining_voids}")
    
    # Test smoothing
    print("Testing smoothing...")
    smoothed_dem = preprocessor.smooth_dem(filled_dem, iterations=2)
    
    print("\n🎉 DEM preprocessing working correctly!")