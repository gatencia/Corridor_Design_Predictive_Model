"""
Coordinate Reference System utilities.
Handles CRS detection, transformation, and UTM zone selection.
"""

import geopandas as gpd
import pyproj
from typing import str, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class CRSUtils:
    """Coordinate Reference System utilities."""
    
    @staticmethod
    def detect_utm_zone(longitude: float) -> str:
        """
        Determine appropriate UTM zone for given longitude.
        
        Parameters:
        -----------
        longitude : float
            Longitude in decimal degrees
            
        Returns:
        --------
        str
            EPSG code for appropriate UTM zone
        """
        # Calculate UTM zone number
        zone_number = int((longitude + 180) / 6) + 1
        
        # For Central Africa, use northern hemisphere UTM zones
        epsg_code = f"EPSG:326{zone_number:02d}"
        
        return epsg_code
    
    @staticmethod
    def get_africa_utm_zones() -> Dict[str, str]:
        """Get common UTM zones for African elephant habitats."""
        return {
            'west_africa': 'EPSG:32628',     # UTM 28N - West Africa
            'central_africa': 'EPSG:32633',  # UTM 33N - Central Africa (Cameroon)
            'east_africa': 'EPSG:32636',     # UTM 36N - East Africa
            'southern_africa': 'EPSG:32735'  # UTM 35S - Southern Africa
        }
    
    @staticmethod
    def get_cameroon_utm_zone() -> str:
        """Get the standard UTM zone for Cameroon."""
        return 'EPSG:32633'  # UTM Zone 33N - covers most of Cameroon
    
    @staticmethod
    def transform_to_utm(gdf: gpd.GeoDataFrame, target_crs: str = None) -> gpd.GeoDataFrame:
        """
        Transform GeoDataFrame to appropriate UTM projection.
        
        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame in geographic coordinates
        target_crs : str, optional
            Target CRS. If None, auto-detect from data extent
            
        Returns:
        --------
        gpd.GeoDataFrame
            GeoDataFrame in UTM projection
        """
        if target_crs is None:
            # Auto-detect UTM zone from data centroid
            bounds = gdf.total_bounds
            center_lon = (bounds[0] + bounds[2]) / 2
            target_crs = CRSUtils.detect_utm_zone(center_lon)
        
        try:
            gdf_utm = gdf.to_crs(target_crs)
            logger.info(f"Transformed to {target_crs}")
            return gdf_utm
        except Exception as e:
            from ..exceptions.validation_errors import CRSTransformationError
            raise CRSTransformationError(f"Failed to transform to {target_crs}: {e}")
    
    @staticmethod
    def get_optimal_crs_for_region(min_lon: float, max_lon: float, 
                                  min_lat: float, max_lat: float) -> str:
        """
        Get optimal CRS for a geographic region.
        
        Parameters:
        -----------
        min_lon, max_lon : float
            Longitude bounds
        min_lat, max_lat : float
            Latitude bounds
            
        Returns:
        --------
        str
            EPSG code for optimal CRS
        """
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        # For Africa, use UTM zones
        if -20 <= center_lat <= 37:  # Africa latitude range
            return CRSUtils.detect_utm_zone(center_lon)
        
        # Fallback to Web Mercator
        return 'EPSG:3857'
    
    @staticmethod
    def validate_crs(crs: str) -> bool:
        """
        Validate that a CRS string is valid.
        
        Parameters:
        -----------
        crs : str
            CRS string (e.g., 'EPSG:4326')
            
        Returns:
        --------
        bool
            True if valid, False otherwise
        """
        try:
            pyproj.CRS(crs)
            return True
        except:
            return False
    
    @staticmethod
    def get_crs_info(crs: str) -> Dict[str, str]:
        """
        Get information about a CRS.
        
        Parameters:
        -----------
        crs : str
            CRS string
            
        Returns:
        --------
        Dict[str, str]
            CRS information
        """
        try:
            crs_obj = pyproj.CRS(crs)
            return {
                'name': crs_obj.name,
                'authority': crs_obj.to_authority()[0] if crs_obj.to_authority() else 'Unknown',
                'code': crs_obj.to_authority()[1] if crs_obj.to_authority() else 'Unknown',
                'units': crs_obj.axis_info[0].unit_name if crs_obj.axis_info else 'Unknown',
                'is_projected': crs_obj.is_projected,
                'is_geographic': crs_obj.is_geographic
            }
        except Exception as e:
            return {'error': str(e)}