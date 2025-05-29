#!/usr/bin/env python3
"""
DEM slope calculation utilities for energy landscape modeling.
Handles slope computation from digital elevation models.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy import ndimage
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class SlopeCalculator:
    """Calculate slope from Digital Elevation Model (DEM)."""
    
    def __init__(self, resolution_m: float = 30.0):
        """
        Initialize slope calculator.
        
        Parameters:
        -----------
        resolution_m : float
            DEM resolution in meters (default 30m for NASADEM)
        """
        self.resolution_m = resolution_m
        
    def calculate_slope_degrees(self, dem_array: np.ndarray, 
                              transform: Affine = None) -> np.ndarray:
        """
        Calculate slope in degrees from DEM array.
        
        Parameters:
        -----------
        dem_array : np.ndarray
            DEM elevation data
        transform : rasterio.transform.Affine, optional
            Raster transform for pixel size calculation
            
        Returns:
        --------
        np.ndarray
            Slope in degrees
        """
        # Get pixel size from transform if available
        if transform is not None:
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
            pixel_size = min(pixel_size_x, pixel_size_y)
        else:
            pixel_size = self.resolution_m
        
        # Handle masked arrays
        if isinstance(dem_array, np.ma.MaskedArray):
            # Fill masked values with a neutral value for gradient calculation
            filled_dem = dem_array.filled(np.nan)
        else:
            filled_dem = dem_array.astype(np.float64)
        
        # Calculate gradients using numpy gradient
        dy, dx = np.gradient(filled_dem, pixel_size)
        
        # Calculate slope in radians
        slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
        
        # Convert to degrees
        slope_degrees = np.degrees(slope_radians)
        
        # Handle NaN values (from masked input or edge effects)
        if isinstance(dem_array, np.ma.MaskedArray):
            slope_degrees = np.ma.array(slope_degrees, mask=dem_array.mask)
        
        logger.debug(f"Calculated slope: {np.nanmin(slope_degrees):.2f}° - {np.nanmax(slope_degrees):.2f}°")
        
        return slope_degrees
    
    def calculate_aspect_degrees(self, dem_array: np.ndarray,
                               transform: Affine = None) -> np.ndarray:
        """
        Calculate aspect in degrees from DEM array.
        
        Aspect is the direction of the steepest slope (0-360°).
        
        Parameters:
        -----------
        dem_array : np.ndarray
            DEM elevation data
        transform : rasterio.transform.Affine, optional
            Raster transform for pixel size calculation
            
        Returns:
        --------
        np.ndarray
            Aspect in degrees (0-360°)
        """
        # Get pixel size from transform if available
        if transform is not None:
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
            pixel_size = min(pixel_size_x, pixel_size_y)
        else:
            pixel_size = self.resolution_m
        
        # Handle masked arrays
        if isinstance(dem_array, np.ma.MaskedArray):
            filled_dem = dem_array.filled(np.nan)
        else:
            filled_dem = dem_array.astype(np.float64)
        
        # Calculate gradients
        dy, dx = np.gradient(filled_dem, pixel_size)
        
        # Calculate aspect in radians
        aspect_radians = np.arctan2(-dy, dx)
        
        # Convert to degrees (0-360°)
        aspect_degrees = np.degrees(aspect_radians)
        aspect_degrees = (aspect_degrees + 360) % 360
        
        # Handle NaN values
        if isinstance(dem_array, np.ma.MaskedArray):
            aspect_degrees = np.ma.array(aspect_degrees, mask=dem_array.mask)
        
        logger.debug(f"Calculated aspect: 0° - 360°")
        
        return aspect_degrees
    
    def calculate_slope_components(self, dem_array: np.ndarray,
                                 transform: Affine = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate slope, aspect, and slope magnitude.
        
        Parameters:
        -----------
        dem_array : np.ndarray
            DEM elevation data
        transform : rasterio.transform.Affine, optional
            Raster transform for pixel size calculation
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Slope (degrees), aspect (degrees), and slope magnitude (rise/run)
        """
        # Get pixel size from transform if available
        if transform is not None:
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
            pixel_size = min(pixel_size_x, pixel_size_y)
        else:
            pixel_size = self.resolution_m
        
        # Handle masked arrays
        if isinstance(dem_array, np.ma.MaskedArray):
            filled_dem = dem_array.filled(np.nan)
            mask = dem_array.mask
        else:
            filled_dem = dem_array.astype(np.float64)
            mask = None
        
        # Calculate gradients
        dy, dx = np.gradient(filled_dem, pixel_size)
        
        # Calculate slope magnitude (rise/run)
        slope_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Calculate slope in degrees
        slope_degrees = np.degrees(np.arctan(slope_magnitude))
        
        # Calculate aspect in degrees
        aspect_radians = np.arctan2(-dy, dx)
        aspect_degrees = np.degrees(aspect_radians)
        aspect_degrees = (aspect_degrees + 360) % 360
        
        # Apply mask if input was masked
        if mask is not None:
            slope_degrees = np.ma.array(slope_degrees, mask=mask)
            aspect_degrees = np.ma.array(aspect_degrees, mask=mask)
            slope_magnitude = np.ma.array(slope_magnitude, mask=mask)
        
        return slope_degrees, aspect_degrees, slope_magnitude
    
    def calculate_slope_horn(self, dem_array: np.ndarray,
                           transform: Affine = None) -> np.ndarray:
        """
        Calculate slope using Horn's method (3x3 moving window).
        
        This method is more robust to noise than simple gradient methods.
        
        Parameters:
        -----------
        dem_array : np.ndarray
            DEM elevation data
        transform : rasterio.transform.Affine, optional
            Raster transform for pixel size calculation
            
        Returns:
        --------
        np.ndarray
            Slope in degrees
        """
        # Get pixel size
        if transform is not None:
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
            pixel_size = min(pixel_size_x, pixel_size_y)
        else:
            pixel_size = self.resolution_m
        
        # Handle masked arrays
        if isinstance(dem_array, np.ma.MaskedArray):
            filled_dem = dem_array.filled(np.nan)
            mask = dem_array.mask
        else:
            filled_dem = dem_array.astype(np.float64)
            mask = None
        
        # Horn's algorithm kernels
        # X-direction (east-west)
        kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]) / (8 * pixel_size)
        
        # Y-direction (north-south)  
        kernel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]]) / (8 * pixel_size)
        
        # Apply convolution
        dx = ndimage.convolve(filled_dem, kernel_x, mode='nearest')
        dy = ndimage.convolve(filled_dem, kernel_y, mode='nearest')
        
        # Calculate slope
        slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_degrees = np.degrees(slope_radians)
        
        # Apply mask if input was masked
        if mask is not None:
            slope_degrees = np.ma.array(slope_degrees, mask=mask)
        
        logger.debug(f"Horn slope calculated: {np.nanmin(slope_degrees):.2f}° - {np.nanmax(slope_degrees):.2f}°")
        
        return slope_degrees
    
    def calculate_slope_statistics(self, slope_array: np.ndarray) -> dict:
        """
        Calculate statistics for slope array.
        
        Parameters:
        -----------
        slope_array : np.ndarray
            Slope data in degrees
            
        Returns:
        --------
        dict
            Slope statistics
        """
        # Handle masked arrays
        if isinstance(slope_array, np.ma.MaskedArray):
            valid_slopes = slope_array.compressed()
        else:
            valid_slopes = slope_array[~np.isnan(slope_array)]
        
        if len(valid_slopes) == 0:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'valid_pixels': 0,
                'total_pixels': slope_array.size
            }
        
        stats = {
            'min': float(np.min(valid_slopes)),
            'max': float(np.max(valid_slopes)),
            'mean': float(np.mean(valid_slopes)),
            'median': float(np.median(valid_slopes)),
            'std': float(np.std(valid_slopes)),
            'valid_pixels': len(valid_slopes),
            'total_pixels': slope_array.size,
            'percentiles': {
                '25': float(np.percentile(valid_slopes, 25)),
                '75': float(np.percentile(valid_slopes, 75)),
                '90': float(np.percentile(valid_slopes, 90)),
                '95': float(np.percentile(valid_slopes, 95))
            }
        }
        
        logger.info(f"Slope statistics: {stats['mean']:.2f}° ± {stats['std']:.2f}° (range: {stats['min']:.2f}° - {stats['max']:.2f}°)")
        
        return stats
    
    def classify_slope_categories(self, slope_array: np.ndarray) -> np.ndarray:
        """
        Classify slopes into categories relevant for elephant movement.
        
        Parameters:
        -----------
        slope_array : np.ndarray
            Slope data in degrees
            
        Returns:
        --------
        np.ndarray
            Slope categories (0=flat, 1=gentle, 2=moderate, 3=steep, 4=very_steep)
        """
        # Define slope categories based on elephant movement literature
        categories = np.zeros_like(slope_array, dtype=np.int8)
        
        # Flat: 0-2°
        categories[(slope_array >= 0) & (slope_array < 2)] = 0
        
        # Gentle: 2-8°
        categories[(slope_array >= 2) & (slope_array < 8)] = 1
        
        # Moderate: 8-15°
        categories[(slope_array >= 8) & (slope_array < 15)] = 2
        
        # Steep: 15-30°
        categories[(slope_array >= 15) & (slope_array < 30)] = 3
        
        # Very steep: 30°+
        categories[slope_array >= 30] = 4
        
        # Handle masked arrays
        if isinstance(slope_array, np.ma.MaskedArray):
            categories = np.ma.array(categories, mask=slope_array.mask)
        
        return categories
    
    def validate_slope_calculation(self, dem_array: np.ndarray, 
                                 slope_array: np.ndarray) -> dict:
        """
        Validate slope calculations against known constraints.
        
        Parameters:
        -----------
        dem_array : np.ndarray
            Original DEM data
        slope_array : np.ndarray
            Calculated slope data
            
        Returns:
        --------
        dict
            Validation results
        """
        validation = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for reasonable slope range
        if isinstance(slope_array, np.ma.MaskedArray):
            valid_slopes = slope_array.compressed()
        else:
            valid_slopes = slope_array[~np.isnan(slope_array)]
        
        if len(valid_slopes) == 0:
            validation['valid'] = False
            validation['issues'].append("No valid slope values calculated")
            return validation
        
        min_slope = np.min(valid_slopes)
        max_slope = np.max(valid_slopes)
        
        # Check minimum slope
        if min_slope < 0:
            validation['issues'].append(f"Negative slopes found: {min_slope:.2f}°")
            validation['valid'] = False
        
        # Check maximum slope
        if max_slope > 90:
            validation['issues'].append(f"Unrealistic steep slopes: {max_slope:.2f}°")
            validation['valid'] = False
        elif max_slope > 60:
            validation['warnings'].append(f"Very steep slopes present: {max_slope:.2f}°")
        
        # Check for flat areas (might indicate processing issues)
        flat_pixels = np.sum(valid_slopes < 0.1)
        flat_percent = (flat_pixels / len(valid_slopes)) * 100
        
        if flat_percent > 50:
            validation['warnings'].append(f"High percentage of flat areas: {flat_percent:.1f}%")
        elif flat_percent > 80:
            validation['issues'].append(f"Excessive flat areas: {flat_percent:.1f}%")
        
        # Check elevation variation in source DEM
        if isinstance(dem_array, np.ma.MaskedArray):
            valid_elevations = dem_array.compressed()
        else:
            valid_elevations = dem_array[~np.isnan(dem_array)]
        
        if len(valid_elevations) > 0:
            elev_range = np.max(valid_elevations) - np.min(valid_elevations)
            if elev_range < 1:
                validation['warnings'].append(f"Low elevation variation: {elev_range:.2f}m")
        
        return validation

# Utility functions
def calculate_slope_from_file(dem_path: str, output_path: str = None,
                            method: str = 'numpy') -> np.ndarray:
    """
    Calculate slope directly from DEM file.
    
    Parameters:
    -----------
    dem_path : str
        Path to DEM file
    output_path : str, optional
        Path to save slope raster
    method : str
        Calculation method ('numpy' or 'horn')
        
    Returns:
    --------
    np.ndarray
        Slope array in degrees
    """
    with rasterio.open(dem_path) as src:
        dem_array = src.read(1, masked=True)
        transform = src.transform
        profile = src.profile.copy()
        
        # Calculate slope
        calc = SlopeCalculator()
        
        if method == 'horn':
            slope_array = calc.calculate_slope_horn(dem_array, transform)
        else:
            slope_array = calc.calculate_slope_degrees(dem_array, transform)
        
        # Save if output path provided
        if output_path:
            profile.update(dtype=rasterio.float32, nodata=-9999)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Handle masked array
                if isinstance(slope_array, np.ma.MaskedArray):
                    slope_data = slope_array.filled(-9999)
                else:
                    slope_data = slope_array
                
                dst.write(slope_data.astype(np.float32), 1)
        
        return slope_array

# Example usage and testing
if __name__ == "__main__":
    # Test slope calculation
    print("📐 Testing Slope Calculation")
    print("=" * 50)
    
    # Create synthetic DEM for testing
    print("Creating synthetic DEM...")
    rows, cols = 100, 100
    x = np.linspace(0, 10, cols)
    y = np.linspace(0, 10, rows)
    X, Y = np.meshgrid(x, y)
    
    # Create terrain with various slopes
    synthetic_dem = 100 + 20 * np.sin(X) + 10 * np.cos(Y) + 5 * X * Y / 50
    
    # Initialize calculator
    calc = SlopeCalculator(resolution_m=30.0)
    
    # Test different methods
    print("\nTesting numpy method...")
    slope_numpy = calc.calculate_slope_degrees(synthetic_dem)
    stats_numpy = calc.calculate_slope_statistics(slope_numpy)
    
    print("\nTesting Horn method...")
    slope_horn = calc.calculate_slope_horn(synthetic_dem)
    stats_horn = calc.calculate_slope_statistics(slope_horn)
    
    # Test slope components
    print("\nTesting slope components...")
    slope, aspect, magnitude = calc.calculate_slope_components(synthetic_dem)
    
    # Test slope classification
    print("\nTesting slope classification...")
    categories = calc.classify_slope_categories(slope_numpy)
    unique_cats, counts = np.unique(categories, return_counts=True)
    
    print("Slope categories:")
    cat_names = ['Flat', 'Gentle', 'Moderate', 'Steep', 'Very Steep']
    for cat, count in zip(unique_cats, counts):
        if cat < len(cat_names):
            print(f"  {cat_names[cat]}: {count} pixels ({count/categories.size*100:.1f}%)")
    
    # Test validation
    print("\nTesting validation...")
    validation = calc.validate_slope_calculation(synthetic_dem, slope_numpy)
    print(f"Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
    
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    
    print("\n🎉 Slope calculation working correctly!")