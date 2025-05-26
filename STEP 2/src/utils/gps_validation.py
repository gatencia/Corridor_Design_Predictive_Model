"""
GPS data validation functions.
Implements coordinate validation, speed filtering, and quality checks.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import logging
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

class GPSValidator:
    """GPS data validation and quality control utilities."""
    
    def __init__(self, config=None):
        """Initialize GPS validator with configuration."""
        self.config = config
        self.validation_results = {}
        
    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate latitude and longitude coordinates.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with location-lat and location-long columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with validation flags added
        """
        logger.info("Validating coordinate bounds...")
        
        # Check for required columns
        required_cols = ['location-lat', 'location-long']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            from ..exceptions.validation_errors import CoordinateValidationError
            raise CoordinateValidationError(f"Missing required columns: {missing_cols}")
        
        # Create validation flags
        df = df.copy()
        df['coord_valid'] = True
        df['coord_issues'] = ''
        
        # Latitude validation (-90 to 90)
        lat_invalid = (df['location-lat'] < -90) | (df['location-lat'] > 90)
        df.loc[lat_invalid, 'coord_valid'] = False
        df.loc[lat_invalid, 'coord_issues'] += 'invalid_lat;'
        
        # Longitude validation (-180 to 180)
        lon_invalid = (df['location-long'] < -180) | (df['location-long'] > 180)
        df.loc[lon_invalid, 'coord_valid'] = False
        df.loc[lon_invalid, 'coord_issues'] += 'invalid_lon;'
        
        # Check for null coordinates
        coord_null = df['location-lat'].isna() | df['location-long'].isna()
        df.loc[coord_null, 'coord_valid'] = False
        df.loc[coord_null, 'coord_issues'] += 'null_coords;'
        
        # Check for exact zero coordinates (often indicates GPS failure)
        zero_coords = (df['location-lat'] == 0) & (df['location-long'] == 0)
        df.loc[zero_coords, 'coord_valid'] = False
        df.loc[zero_coords, 'coord_issues'] += 'zero_coords;'
        
        # Regional validation for Central Africa (rough bounds)
        # Cameroon approximate bounds: lat 1-13, lon 8-17
        africa_bounds = (
            (df['location-lat'] < -10) | (df['location-lat'] > 20) |
            (df['location-long'] < 5) | (df['location-long'] > 30)
        )
        df.loc[africa_bounds, 'coord_issues'] += 'outside_africa;'
        
        # Store validation results
        n_invalid = lat_invalid.sum() + lon_invalid.sum() + coord_null.sum()
        n_zero = zero_coords.sum()
        n_outside = africa_bounds.sum()
        
        self.validation_results['coordinates'] = {
            'total_points': len(df),
            'invalid_coordinates': int(n_invalid),
            'zero_coordinates': int(n_zero), 
            'outside_africa_bounds': int(n_outside),
            'valid_coordinates': int((~lat_invalid & ~lon_invalid & ~coord_null).sum())
        }
        
        logger.info(f"Coordinate validation: {self.validation_results['coordinates']['valid_coordinates']}/{len(df)} valid")
        
        return df
    
    def calculate_movement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate movement speeds and distances between consecutive GPS fixes.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with coordinates and timestamps
            
        Returns:  
        --------
        pd.DataFrame
            DataFrame with movement metrics added
        """
        logger.info("Calculating movement metrics...")
        
        df = df.copy()
        df = df.sort_values(['individual-local-identifier', 'timestamp'])
        
        # Initialize movement columns
        df['distance_km'] = np.nan
        df['time_diff_hours'] = np.nan
        df['speed_kmh'] = np.nan
        df['movement_valid'] = True
        df['movement_issues'] = ''
        
        # Group by individual to calculate movements
        for individual_id in df['individual-local-identifier'].unique():
            mask = df['individual-local-identifier'] == individual_id
            individual_data = df[mask].copy()
            
            if len(individual_data) < 2:
                continue
                
            # Calculate distances and time differences
            distances = []
            time_diffs = []
            speeds = []
            
            for i in range(1, len(individual_data)):
                prev_row = individual_data.iloc[i-1]
                curr_row = individual_data.iloc[i]
                
                # Calculate distance using geodesic distance
                try:
                    coord1 = (prev_row['location-lat'], prev_row['location-long'])
                    coord2 = (curr_row['location-lat'], curr_row['location-long'])
                    distance_km = geodesic(coord1, coord2).kilometers
                    distances.append(distance_km)
                except:
                    distances.append(np.nan)
                
                # Calculate time difference
                try:
                    time_diff = curr_row['timestamp'] - prev_row['timestamp']
                    time_diff_hours = time_diff.total_seconds() / 3600
                    time_diffs.append(time_diff_hours)
                    
                    # Calculate speed
                    if time_diff_hours > 0 and not np.isnan(distances[-1]):
                        speed = distances[-1] / time_diff_hours
                        speeds.append(speed)
                    else:
                        speeds.append(np.nan)
                except:
                    time_diffs.append(np.nan)
                    speeds.append(np.nan)
            
            # Update DataFrame
            indices = individual_data.index[1:]  # Skip first row
            df.loc[indices, 'distance_km'] = distances
            df.loc[indices, 'time_diff_hours'] = time_diffs
            df.loc[indices, 'speed_kmh'] = speeds
        
        # Validate movement speeds
        max_speed = getattr(self.config.processing, 'gps_speed_filter_kmh', 100.0) if self.config else 100.0
        high_speed = df['speed_kmh'] > max_speed
        df.loc[high_speed, 'movement_valid'] = False
        df.loc[high_speed, 'movement_issues'] += f'high_speed(>{max_speed}kmh);'
        
        # Flag very large time gaps
        max_gap_hours = getattr(self.config.processing, 'gps_max_gap_hours', 24.0) if self.config else 24.0
        large_gaps = df['time_diff_hours'] > max_gap_hours
        df.loc[large_gaps, 'movement_issues'] += f'large_gap(>{max_gap_hours}h);'
        
        # Store validation results
        self.validation_results['movement'] = {
            'high_speed_fixes': int(high_speed.sum()),
            'large_time_gaps': int(large_gaps.sum()),
            'max_speed_observed': float(df['speed_kmh'].max()) if not df['speed_kmh'].isna().all() else 0,
            'median_speed': float(df['speed_kmh'].median()) if not df['speed_kmh'].isna().all() else 0
        }
        
        logger.info(f"Movement validation: {high_speed.sum()} high-speed fixes flagged")
        
        return df
    
    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle duplicate GPS fixes.
        
        Parameters:
        -----------
        df : pd.DataFrame
            GPS data DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with duplicate flags
        """
        logger.info("Detecting duplicate GPS fixes...")
        
        df = df.copy()
        
        # Define columns for duplicate detection
        duplicate_cols = ['individual-local-identifier', 'timestamp', 'location-lat', 'location-long']
        
        # Mark duplicates
        df['is_duplicate'] = df.duplicated(subset=duplicate_cols, keep='first')
        
        # Mark near-duplicates (same location within time threshold)
        df['is_near_duplicate'] = False
        min_interval = getattr(self.config.processing, 'gps_min_interval_minutes', 5) if self.config else 5
        
        for individual_id in df['individual-local-identifier'].unique():
            mask = df['individual-local-identifier'] == individual_id
            individual_data = df[mask].sort_values('timestamp')
            
            for i in range(1, len(individual_data)):
                curr_idx = individual_data.index[i]
                prev_idx = individual_data.index[i-1]
                
                curr_row = individual_data.iloc[i]
                prev_row = individual_data.iloc[i-1]
                
                # Check time difference
                time_diff_min = (curr_row['timestamp'] - prev_row['timestamp']).total_seconds() / 60
                
                # Check coordinate similarity (within 0.001 degrees ≈ 100m)
                coord_similar = (
                    abs(curr_row['location-lat'] - prev_row['location-lat']) < 0.001 and
                    abs(curr_row['location-long'] - prev_row['location-long']) < 0.001
                )
                
                if time_diff_min < min_interval and coord_similar:
                    df.loc[curr_idx, 'is_near_duplicate'] = True
        
        # Store validation results
        n_duplicates = df['is_duplicate'].sum()
        n_near_duplicates = df['is_near_duplicate'].sum()
        
        self.validation_results['duplicates'] = {
            'exact_duplicates': int(n_duplicates),
            'near_duplicates': int(n_near_duplicates),
            'total_flagged': int(n_duplicates + n_near_duplicates)
        }
        
        logger.info(f"Duplicate detection: {n_duplicates} exact, {n_near_duplicates} near-duplicates")
        
        return df