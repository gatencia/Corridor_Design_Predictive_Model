"""
Geometric operations for AOI generation.
Implements convex hull, alpha shapes, and buffering operations.
"""

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from typing import Union

def create_convex_hull(gdf: gpd.GeoDataFrame) -> Polygon:
    """Create convex hull from GPS points."""
    pass

def create_alpha_shape(gdf: gpd.GeoDataFrame, alpha: float = 0.1) -> Union[Polygon, MultiPolygon]:
    """Create concave hull (alpha shape) from GPS points."""
    pass

def buffer_geometry(geometry, distance_m: float, crs: str) -> Union[Polygon, MultiPolygon]:
    """Apply buffer to geometry in projected coordinates."""
    pass
