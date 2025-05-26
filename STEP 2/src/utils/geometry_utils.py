"""
Geometric operations for AOI generation.
Implements convex hull, alpha shapes, and buffering operations.
"""

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from typing import Union
import pyproj
import logging

logger = logging.getLogger(__name__)

class GeometryUtils:
    """Geometric operations for AOI generation."""
    
    @staticmethod
    def create_convex_hull(gdf: gpd.GeoDataFrame) -> Polygon:
        """
        Create convex hull from GPS points.
        
        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with point geometries
            
        Returns:
        --------
        Polygon
            Convex hull polygon
        """
        if len(gdf) == 0:
            raise ValueError("Cannot create hull from empty GeoDataFrame")
        
        # Get union of all points and create convex hull
        points_union = unary_union(gdf.geometry)
        hull = points_union.convex_hull
        
        # Ensure we return a Polygon (not Point or LineString for small datasets)
        if isinstance(hull, Point):
            # Single point - create small circular buffer
            hull = hull.buffer(0.001)  # ~100m buffer
        elif hasattr(hull, 'geom_type') and hull.geom_type == 'LineString':
            # Linear arrangement - create small buffer
            hull = hull.buffer(0.001)
        
        return hull
    
    @staticmethod
    def create_alpha_shape(gdf: gpd.GeoDataFrame, alpha: float = 0.1) -> Union[Polygon, MultiPolygon]:
        """
        Create concave hull (alpha shape) from GPS points.
        Note: This is a simplified implementation. For production use,
        consider using the alpha_shape library or similar.
        
        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with point geometries
        alpha : float
            Alpha parameter controlling concaveness
            
        Returns:
        --------
        Union[Polygon, MultiPolygon]
            Alpha shape polygon(s)
        """
        # For now, return convex hull as placeholder
        # TODO: Implement proper alpha shape algorithm
        logger.warning("Alpha shape not fully implemented, using convex hull")
        return GeometryUtils.create_convex_hull(gdf)
    
    @staticmethod 
    def buffer_geometry(geometry: Union[Polygon, MultiPolygon], 
                       distance_m: float, 
                       crs: str) -> Union[Polygon, MultiPolygon]:
        """
        Apply buffer to geometry in projected coordinates.
        
        Parameters:
        -----------
        geometry : Polygon or MultiPolygon
            Input geometry
        distance_m : float
            Buffer distance in meters
        crs : str
            Coordinate reference system of the geometry
            
        Returns:
        --------
        Polygon or MultiPolygon
            Buffered geometry
        """
        # Check if CRS is projected (should have meters as units)
        try:
            crs_info = pyproj.CRS(crs)
            if not crs_info.is_projected:
                logger.warning(f"CRS {crs} is not projected, buffer may be inaccurate")
        except:
            logger.warning(f"Could not validate CRS {crs}")
        
        buffered = geometry.buffer(distance_m)
        return buffered
    
    @staticmethod
    def calculate_area_km2(geometry: Union[Polygon, MultiPolygon], crs: str) -> float:
        """
        Calculate area of geometry in square kilometers.
        
        Parameters:
        -----------
        geometry : Polygon or MultiPolygon
            Input geometry
        crs : str
            Coordinate reference system (should be projected)
            
        Returns:
        --------
        float
            Area in square kilometers
        """
        try:
            crs_info = pyproj.CRS(crs)
            if not crs_info.is_projected:
                logger.warning(f"CRS {crs} is not projected, area calculation may be inaccurate")
                # Convert to approximate equal-area projection
                # This is a rough approximation
                return geometry.area * 111 * 111  # Rough conversion from degrees² to km²
        except:
            logger.warning(f"Could not validate CRS {crs}")
        
        # For projected coordinates, assume units are meters
        area_m2 = geometry.area
        area_km2 = area_m2 / 1e6  # Convert m² to km²
        
        return area_km2
    
    @staticmethod
    def simplify_geometry(geometry: Union[Polygon, MultiPolygon], 
                         tolerance_m: float = 10.0) -> Union[Polygon, MultiPolygon]:
        """
        Simplify geometry to reduce complexity.
        
        Parameters:
        -----------
        geometry : Polygon or MultiPolygon
            Input geometry
        tolerance_m : float
            Simplification tolerance in meters
            
        Returns:
        --------
        Polygon or MultiPolygon
            Simplified geometry
        """
        simplified = geometry.simplify(tolerance_m, preserve_topology=True)
        return simplified
    
    @staticmethod
    def validate_geometry(geometry: Union[Polygon, MultiPolygon]) -> bool:
        """
        Validate that geometry is valid and not empty.
        
        Parameters:
        -----------
        geometry : Polygon or MultiPolygon
            Geometry to validate
            
        Returns:
        --------
        bool
            True if valid, False otherwise
        """
        try:
            return (geometry is not None and 
                   not geometry.is_empty and 
                   geometry.is_valid and
                   geometry.area > 0)
        except:
            return False
    
    @staticmethod
    def fix_geometry(geometry: Union[Polygon, MultiPolygon]) -> Union[Polygon, MultiPolygon]:
        """
        Attempt to fix invalid geometry.
        
        Parameters:
        -----------
        geometry : Polygon or MultiPolygon
            Potentially invalid geometry
            
        Returns:
        --------
        Polygon or MultiPolygon
            Fixed geometry
        """
        if GeometryUtils.validate_geometry(geometry):
            return geometry
        
        try:
            # Try buffer(0) to fix self-intersections
            fixed = geometry.buffer(0)
            if GeometryUtils.validate_geometry(fixed):
                logger.info("Fixed geometry using buffer(0)")
                return fixed
        except:
            pass
        
        try:
            # Try convex hull as last resort
            fixed = geometry.convex_hull
            if GeometryUtils.validate_geometry(fixed):
                logger.warning("Used convex hull to fix invalid geometry")
                return fixed
        except:
            pass
        
        logger.error("Could not fix invalid geometry")
        raise ValueError("Cannot fix invalid geometry")
    
    @staticmethod
    def get_geometry_bounds(geometry: Union[Polygon, MultiPolygon]) -> tuple:
        """
        Get bounding box of geometry.
        
        Parameters:
        -----------
        geometry : Polygon or MultiPolygon
            Input geometry
            
        Returns:
        --------
        tuple
            (minx, miny, maxx, maxy)
        """
        return geometry.bounds
    
    @staticmethod
    def get_geometry_centroid(geometry: Union[Polygon, MultiPolygon]) -> Point:
        """
        Get centroid of geometry.
        
        Parameters:
        -----------
        geometry : Polygon or MultiPolygon
            Input geometry
            
        Returns:
        --------
        Point
            Centroid point
        """
        return geometry.centroid