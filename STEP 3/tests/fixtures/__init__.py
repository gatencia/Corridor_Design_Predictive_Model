"""Test fixtures for EnergyScape testing"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

def create_synthetic_dem(rows: int = 100, cols: int = 100, 
                        base_elevation: float = 100.0) -> np.ndarray:
    """Create synthetic DEM for testing."""
    x = np.linspace(0, 10, cols)
    y = np.linspace(0, 10, rows)
    X, Y = np.meshgrid(x, y)
    
    # Create varied terrain with different features
    dem = (base_elevation + 
           20 * np.sin(X) +           # E-W ridges
           15 * np.cos(Y) +           # N-S valleys  
           10 * np.sin(2 * X) * np.cos(2 * Y) +  # Cross pattern
           5 * np.random.random((rows, cols)))    # Noise
    
    return dem

def create_test_energy_surface(rows: int = 50, cols: int = 50) -> np.ndarray:
    """Create synthetic energy surface for testing."""
    # Base on synthetic slope pattern
    dem = create_synthetic_dem(rows, cols)
    
    # Simple slope calculation
    dy, dx = np.gradient(dem, 30.0)  # 30m resolution
    slopes = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Simple energy cost (mass=4000kg, basic Pontzer equation)
    base_cost = 1.0 * (4000 ** 0.75)  # α × M^β
    slope_multiplier = 1 + 1.5 * np.abs(np.sin(np.radians(slopes)))  # 1 + γ × |sin(θ)|
    energy_surface = base_cost * slope_multiplier
    
    return energy_surface

__all__ = [
    'create_synthetic_dem',
    'create_test_energy_surface'
]
