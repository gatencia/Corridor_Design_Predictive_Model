"""
Generate synthetic GPS data for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_track(n_points: int = 100, 
                         start_lat: float = 8.0, 
                         start_lon: float = 13.0) -> pd.DataFrame:
    """Generate synthetic GPS track for testing."""
    
    # Generate random walk
    np.random.seed(42)
    lat_steps = np.random.normal(0, 0.01, n_points)
    lon_steps = np.random.normal(0, 0.01, n_points)
    
    # Create coordinate sequences
    lats = np.cumsum(lat_steps) + start_lat
    lons = np.cumsum(lon_steps) + start_lon
    
    # Create timestamps
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'location-lat': lats,
        'location-long': lons,
        'tag-local-identifier': ['TEST001'] * n_points,
        'individual-local-identifier': [1] * n_points
    })
