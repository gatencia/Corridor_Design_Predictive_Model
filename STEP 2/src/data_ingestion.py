#!/usr/bin/env python3
"""
Main GPS data processing module for elephant corridor analysis.
Handles CSV parsing, validation, and GeoPandas conversion.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import logging
import warnings
from tqdm import tqdm
import json

# Geometric operations
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import pyproj
from geopy.distance import geodesic

# Import utilities - using absolute imports
try:
    from utils.gps_validation import GPSValidator
    from utils.crs_utils import CRSUtils
    from utils.geometry_utils import GeometryUtils
    from exceptions.validation_errors import (
        GPSValidationError, CoordinateValidationError, 
        TemporalValidationError, FileFormatError
    )
except ImportError:
    # Fallback to try relative imports
    try:
        from .utils.gps_validation import GPSValidator
        from .utils.crs_utils import CRSUtils
        from .utils.geometry_utils import GeometryUtils
        from .exceptions.validation_errors import (
            GPSValidationError, CoordinateValidationError, 
            TemporalValidationError, FileFormatError
        )
    except ImportError:
        # Define minimal fallback classes
        class GPSValidator:
            def __init__(self, config=None):
                self.config = config
                self.validation_results = {}
            def validate_coordinates(self, df):
                df['coord_valid'] = True
                return df
            def calculate_movement_metrics(self, df):
                return df
            def detect_duplicates(self, df):
                df['is_duplicate'] = False
                return df
        
        class CRSUtils:
            @staticmethod
            def detect_utm_zone(longitude):
                zone_number = int((longitude + 180) / 6) + 1
                return f"EPSG:326{zone_number:02d}"
            @staticmethod
            def transform_to_utm(gdf, target_crs=None):
                if target_crs is None:
                    bounds = gdf.total_bounds
                    center_lon = (bounds[0] + bounds[2]) / 2
                    target_crs = CRSUtils.detect_utm_zone(center_lon)
                return gdf.to_crs(target_crs)
        
        class GeometryUtils:
            @staticmethod
            def create_convex_hull(gdf):
                return unary_union(gdf.geometry).convex_hull
            @staticmethod
            def buffer_geometry(geometry, distance_m, crs):
                return geometry.buffer(distance_m)
        
        class GPSValidationError(Exception):
            pass
        class CoordinateValidationError(GPSValidationError):
            pass
        class TemporalValidationError(GPSValidationError):
            pass
        class FileFormatError(GPSValidationError):
            pass

# Configuration integration
try:
    from config.project_config import get_config
except ImportError:
    # Fallback configuration
    class MockConfig:
        def __init__(self):
            self.gps_speed_filter_kmh = 100.0
            self.gps_min_interval_minutes = 5
            self.gps_max_gap_hours = 24
            self.default_buffer_distance_m = 5000.0
            self.default_projected_crs = "EPSG:32633"
            self.default_geographic_crs = "EPSG:4326"
    
    def get_config():
        return type('Config', (), {'processing': MockConfig(), 'gis': MockConfig()})()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPSDataProcessor:
    """Main class for processing GPS collar data."""
    
    def __init__(self, config=None):
        """Initialize GPS data processor."""
        self.config = config or get_config()
        self.validator = GPSValidator(self.config)
        self.data_quality_report = {}
        
    def load_gps_data(self, file_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Load and validate GPS data from CSV file.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to GPS CSV file
            
        Returns:
        --------
        gpd.GeoDataFrame
            Processed GPS data
        """
        file_path = Path(file_path)
        logger.info(f"Loading GPS data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"GPS file not found: {file_path}")
        
        try:
            # Load CSV data
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} GPS fixes from {file_path.name}")
            
            # Validate expected columns
            expected_cols = ['timestamp', 'location-lat', 'location-long', 
                           'tag-local-identifier', 'individual-local-identifier']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            
            if missing_cols:
                raise FileFormatError(f"Missing required columns: {missing_cols}")
            
            # Parse timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add source file information
            df['source_file'] = file_path.name
            
            # Validate data quality
            df = self.validator.validate_coordinates(df)
            df = self.validator.calculate_movement_metrics(df)
            df = self.validator.detect_duplicates(df)
            
            # Create GeoDataFrame
            gdf = self._create_geodataframe(df)
            
            # Store data quality metrics
            coord_valid = getattr(self.validator, 'validation_results', {}).get('coordinates', {}).get('valid_coordinates', len(df))
            movement_high_speed = getattr(self.validator, 'validation_results', {}).get('movement', {}).get('high_speed_fixes', 0)
            duplicates_total = getattr(self.validator, 'validation_results', {}).get('duplicates', {}).get('total_flagged', 0)
            
            self.data_quality_report[file_path.name] = {
                'total_fixes': len(df),
                'valid_coordinates': coord_valid,
                'high_speed_fixes': movement_high_speed,
                'duplicates': duplicates_total,
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'individuals': df['individual-local-identifier'].nunique(),
                'collar_ids': df['tag-local-identifier'].nunique()
            }
            
            logger.info(f"Successfully processed {len(gdf)} GPS fixes")
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading GPS data from {file_path}: {e}")
            raise
    
    def process_multiple_files(self, file_paths: List[Union[str, Path]], 
                             output_dir: Union[str, Path] = None) -> gpd.GeoDataFrame:
        """
        Process multiple GPS collar files.
        
        Parameters:
        -----------
        file_paths : List[str or Path]
            List of GPS CSV file paths
        output_dir : str or Path, optional
            Directory to save intermediate results
            
        Returns:
        --------
        gpd.GeoDataFrame
            Combined GPS data from all files
        """
        logger.info(f"Processing {len(file_paths)} GPS files...")
        
        all_gdfs = []
        
        for file_path in tqdm(file_paths, desc="Processing GPS files"):
            try:
                gdf = self.load_gps_data(file_path)
                all_gdfs.append(gdf)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        if not all_gdfs:
            raise ValueError("No GPS files were successfully processed")
        
        # Combine all GeoDataFrames
        logger.info("Combining GPS data from all files...")
        combined_gdf = gpd.pd.concat(all_gdfs, ignore_index=True)
        
        # Sort by individual and timestamp
        combined_gdf = combined_gdf.sort_values(['individual-local-identifier', 'timestamp'])
        
        # Remove duplicates across files
        logger.info("Removing cross-file duplicates...")
        duplicate_cols = ['individual-local-identifier', 'timestamp', 'location-lat', 'location-long']
        combined_gdf = combined_gdf.drop_duplicates(subset=duplicate_cols, keep='first')
        
        logger.info(f"Combined dataset: {len(combined_gdf)} GPS fixes from {combined_gdf['individual-local-identifier'].nunique()} individuals")
        
        # Save combined data if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"gps_tracks_combined_{timestamp}.gpq"
            
            combined_gdf.to_parquet(output_file)
            logger.info(f"Saved combined GPS data to {output_file}")
            
            # Save data quality report
            report_file = output_dir / f"data_quality_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(self.data_quality_report, f, indent=2)
            logger.info(f"Saved data quality report to {report_file}")
        
        return combined_gdf
    
    def generate_aoi(self, gdf: gpd.GeoDataFrame, 
                    buffer_km: float = 5.0,
                    method: str = 'convex_hull') -> gpd.GeoDataFrame:
        """
        Generate Area of Interest polygon from GPS points.
        
        Parameters:
        -----------
        gdf : gpd.GeoDataFrame
            GPS tracking data
        buffer_km : float
            Buffer distance in kilometers
        method : str
            Method for AOI generation ('convex_hull' or 'alpha_shape')
            
        Returns:
        --------
        gpd.GeoDataFrame
            AOI polygon as GeoDataFrame
        """
        logger.info(f"Generating AOI with {buffer_km}km buffer using {method} method")
        
        if len(gdf) == 0:
            raise ValueError("Cannot generate AOI from empty GeoDataFrame")
        
        # Ensure data is in geographic coordinates for initial hull generation
        if gdf.crs != 'EPSG:4326':
            gdf_geo = gdf.to_crs('EPSG:4326')
        else:
            gdf_geo = gdf.copy()
        
        # Generate initial hull
        if method == 'convex_hull':
            hull_geo = GeometryUtils.create_convex_hull(gdf_geo)
        elif method == 'alpha_shape':
            hull_geo = GeometryUtils.create_alpha_shape(gdf_geo)
        else:
            raise ValueError(f"Unknown AOI method: {method}")
        
        # Transform to appropriate UTM for buffering
        bounds = gdf_geo.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        utm_crs = CRSUtils.detect_utm_zone(center_lon)
        
        # Create temporary GeoDataFrame for transformation
        hull_gdf = gpd.GeoDataFrame([1], geometry=[hull_geo], crs='EPSG:4326')
        hull_utm = hull_gdf.to_crs(utm_crs)
        
        # Apply buffer in meters
        buffer_m = buffer_km * 1000
        buffered_geom = GeometryUtils.buffer_geometry(
            hull_utm.geometry.iloc[0], buffer_m, utm_crs
        )
        
        # Create final AOI GeoDataFrame
        aoi_gdf = gpd.GeoDataFrame(
            [{
                'aoi_id': 1,
                'method': method,
                'buffer_km': buffer_km,
                'utm_crs': utm_crs,
                'n_gps_points': len(gdf),
                'individuals': gdf['individual-local-identifier'].nunique(),
                'date_range_start': gdf['timestamp'].min(),
                'date_range_end': gdf['timestamp'].max(),
                'area_km2': buffered_geom.area / 1e6  # Convert m² to km²
            }],
            geometry=[buffered_geom],
            crs=utm_crs
        )
        
        logger.info(f"Generated AOI: {aoi_gdf['area_km2'].iloc[0]:.1f} km²")
        
        return aoi_gdf
    
    def _create_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert DataFrame to GeoDataFrame with Point geometries."""
        
        # Filter valid coordinates if validation was done
        if 'coord_valid' in df.columns:
            valid_coords = df['coord_valid'] == True
            df_clean = df[valid_coords].copy()
        else:
            df_clean = df.copy()
        
        if len(df_clean) == 0:
            raise ValueError("No valid coordinates found in dataset")
        
        # Create Point geometries
        geometry = [Point(lon, lat) for lon, lat in 
                   zip(df_clean['location-long'], df_clean['location-lat'])]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs='EPSG:4326')
        
        return gdf
    
    def export_results(self, gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame, 
                      output_dir: Union[str, Path], study_name: str = "elephant_study"):
        """
        Export processed GPS data and AOI to various formats.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export GPS tracks
        gps_file = output_dir / f"gps_tracks_{study_name}_{timestamp}.gpq"
        gdf.to_parquet(gps_file)
        logger.info(f"Exported GPS tracks to {gps_file}")
        
        # CSV (for compatibility)
        csv_file = output_dir / f"gps_tracks_{study_name}_{timestamp}.csv"
        df_export = gdf.drop(columns=['geometry']).copy()
        df_export['longitude'] = gdf.geometry.x
        df_export['latitude'] = gdf.geometry.y
        df_export.to_csv(csv_file, index=False)
        logger.info(f"Exported GPS tracks to {csv_file}")
        
        # Export AOI
        buffer_km = aoi_gdf['buffer_km'].iloc[0]
        
        # GeoJSON (web-friendly)
        aoi_geojson = output_dir / f"aoi_{study_name}_{buffer_km}km_{timestamp}.geojson"
        aoi_gdf.to_file(aoi_geojson, driver='GeoJSON')
        logger.info(f"Exported AOI to {aoi_geojson}")
        
        # Shapefile (GIS standard)
        aoi_shp = output_dir / f"aoi_{study_name}_{buffer_km}km_{timestamp}.shp"
        aoi_gdf.to_file(aoi_shp)
        logger.info(f"Exported AOI to {aoi_shp}")
        
        # Summary statistics
        summary = {
            'processing_timestamp': timestamp,
            'study_name': study_name,
            'gps_data': {
                'total_fixes': len(gdf),
                'individuals': int(gdf['individual-local-identifier'].nunique()),
                'collar_ids': list(gdf['tag-local-identifier'].unique()),
                'date_range': {
                    'start': gdf['timestamp'].min().isoformat(),
                    'end': gdf['timestamp'].max().isoformat()
                },
                'spatial_extent': {
                    'min_lat': float(gdf['location-lat'].min()),
                    'max_lat': float(gdf['location-lat'].max()),
                    'min_lon': float(gdf['location-long'].min()),
                    'max_lon': float(gdf['location-long'].max())
                }
            },
            'aoi': {
                'method': aoi_gdf['method'].iloc[0],
                'buffer_km': float(aoi_gdf['buffer_km'].iloc[0]),
                'area_km2': float(aoi_gdf['area_km2'].iloc[0]),
                'utm_crs': aoi_gdf['utm_crs'].iloc[0]
            },
            'data_quality': self.data_quality_report,
            'files_exported': {
                'gps_parquet': str(gps_file),
                'gps_csv': str(csv_file),
                'aoi_geojson': str(aoi_geojson),
                'aoi_shapefile': str(aoi_shp)
            }
        }
        
        summary_file = output_dir / f"processing_summary_{study_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Exported processing summary to {summary_file}")


def main():
    """Example usage with your Cameroon elephant data."""
    
    # Path to your GPS data directory
    data_dir = Path("../GPS_Collar_CSV_Mark")
    
    # Find all CSV files in the directory
    gps_files = list(data_dir.glob("*.csv"))
    
    if not gps_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return
    
    logger.info(f"Found {len(gps_files)} GPS CSV files")
    for file in gps_files[:5]:  # Show first 5 files
        logger.info(f"  - {file.name}")
    if len(gps_files) > 5:
        logger.info(f"  ... and {len(gps_files) - 5} more files")
    
    try:
        # Initialize processor
        processor = GPSDataProcessor()
        
        # Process all files
        logger.info("Starting GPS data processing pipeline...")
        
        combined_gdf = processor.process_multiple_files(
            gps_files, 
            output_dir="data/processed"
        )
        
        # Generate AOI
        aoi_gdf = processor.generate_aoi(
            combined_gdf, 
            buffer_km=5.0, 
            method='convex_hull'
        )
        
        # Export results
        processor.export_results(
            combined_gdf, 
            aoi_gdf, 
            output_dir="data/outputs",
            study_name="cameroon_elephants"
        )
        
        logger.info("GPS data processing completed successfully!")
        
        # Print summary
        print(f"\n🐘 GPS Data Processing Summary")
        print(f"={'='*50}")
        print(f"Total GPS fixes: {len(combined_gdf):,}")
        print(f"Individuals tracked: {combined_gdf['individual-local-identifier'].nunique()}")
        print(f"Date range: {combined_gdf['timestamp'].min().date()} to {combined_gdf['timestamp'].max().date()}")
        print(f"AOI area: {aoi_gdf['area_km2'].iloc[0]:.1f} km²")
        print(f"Study region: {aoi_gdf['utm_crs'].iloc[0]}")
        
    except Exception as e:
        logger.error(f"GPS processing failed: {e}")
        raise


if __name__ == "__main__":
    main()