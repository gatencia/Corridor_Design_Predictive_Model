#!/usr/bin/env python3
"""
Step 2: Data Ingestion Module - Directory Structure Generator
Creates the complete folder structure and placeholder files for GPS data processing.
"""

import os
from pathlib import Path
from datetime import datetime

def create_step2_structure():
    """
    Creates the directory structure and placeholder files for Step 2: Data Ingestion Module
    """
    
    # Define the project root
    project_root = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/STEP 2")
    
    # Define directory structure
    directories = [
        # Source code directories
        project_root / "src",
        project_root / "src" / "utils",
        project_root / "src" / "exceptions",
        
        # Testing directories
        project_root / "tests",
        project_root / "tests" / "fixtures",
        project_root / "tests" / "data",
        
        # Documentation
        project_root / "docs",
        
        # Notebooks
        project_root / "notebooks",
        
        # Data directories
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "outputs",
        
        # Logs
        project_root / "logs"
    ]
    
    # Define files to create with their paths and content
    files = {
        # Source code files
        project_root / "src" / "__init__.py": '"""Elephant Corridor Analysis - Source Package"""',
        
        project_root / "src" / "data_ingestion.py": '''"""
Main GPS data processing module for elephant corridor analysis.
Handles CSV parsing, validation, and GeoPandas conversion.
"""

from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging

# Placeholder for main data ingestion class
class GPSDataProcessor:
    """Main class for processing GPS collar data."""
    
    def __init__(self, config=None):
        """Initialize GPS data processor."""
        pass
    
    def load_gps_data(self, file_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """Load and validate GPS data from CSV file."""
        pass
    
    def process_multiple_files(self, file_paths: List[Union[str, Path]]) -> gpd.GeoDataFrame:
        """Process multiple GPS collar files."""
        pass
    
    def generate_aoi(self, gdf: gpd.GeoDataFrame, buffer_km: float = 5.0) -> gpd.GeoDataFrame:
        """Generate Area of Interest polygon from GPS points."""
        pass
''',
        
        project_root / "src" / "utils" / "__init__.py": '"""Utility modules for GPS data processing"""',
        
        project_root / "src" / "utils" / "gps_validation.py": '''"""
GPS data validation functions.
Implements coordinate validation, speed filtering, and quality checks.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Validate latitude and longitude coordinates."""
    pass

def calculate_speed_filter(df: pd.DataFrame, max_speed_kmh: float = 100.0) -> pd.DataFrame:
    """Filter out unrealistic movement speeds."""
    pass

def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and handle duplicate GPS fixes."""
    pass
''',
        
        project_root / "src" / "utils" / "crs_utils.py": '''"""
Coordinate Reference System utilities.
Handles CRS detection, transformation, and UTM zone selection.
"""

import geopandas as gpd
import pyproj
from typing import str, Tuple

def detect_utm_zone(longitude: float) -> str:
    """Determine appropriate UTM zone for given longitude."""
    pass

def transform_to_utm(gdf: gpd.GeoDataFrame, target_crs: str = None) -> gpd.GeoDataFrame:
    """Transform GeoDataFrame to appropriate UTM projection."""
    pass

def get_africa_utm_zones() -> dict:
    """Get common UTM zones for African elephant habitats."""
    pass
''',
        
        project_root / "src" / "utils" / "geometry_utils.py": '''"""
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
''',
        
        project_root / "src" / "exceptions" / "__init__.py": '"""Custom exception classes for GPS data processing"""',
        
        project_root / "src" / "exceptions" / "validation_errors.py": '''"""
Custom exception classes for data validation errors.
"""

class GPSValidationError(Exception):
    """Base exception for GPS data validation errors."""
    pass

class CoordinateValidationError(GPSValidationError):
    """Exception for invalid coordinate values."""
    pass

class TemporalValidationError(GPSValidationError):
    """Exception for temporal data issues."""
    pass

class FileFormatError(GPSValidationError):
    """Exception for file format issues."""
    pass

class CRSTransformationError(GPSValidationError):
    """Exception for CRS transformation issues."""
    pass
''',
        
        # Test files
        project_root / "tests" / "__init__.py": '"""Test package for elephant corridor analysis"""',
        
        project_root / "tests" / "test_data_ingestion.py": '''"""
Unit tests for GPS data ingestion module.
"""

import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
import tempfile

# Placeholder test class
class TestGPSDataProcessor:
    """Test suite for GPS data processing functionality."""
    
    def test_load_gps_data(self):
        """Test loading GPS data from CSV file."""
        pass
    
    def test_coordinate_validation(self):
        """Test coordinate validation functions."""
        pass
    
    def test_speed_filtering(self):
        """Test speed-based filtering."""
        pass
    
    def test_aoi_generation(self):
        """Test Area of Interest generation."""
        pass
''',
        
        project_root / "tests" / "test_gps_validation.py": '''"""
Unit tests for GPS validation functions.
"""

import pytest
import numpy as np
import pandas as pd

class TestGPSValidation:
    """Test suite for GPS validation functions."""
    
    def test_coordinate_bounds(self):
        """Test coordinate boundary validation."""
        pass
    
    def test_speed_calculation(self):
        """Test speed calculation and filtering."""
        pass
    
    def test_duplicate_detection(self):
        """Test duplicate GPS fix detection."""
        pass
''',
        
        project_root / "tests" / "fixtures" / "__init__.py": '"""Test fixtures for GPS data testing"""',
        
        project_root / "tests" / "fixtures" / "sample_gps_data.py": '''"""
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
''',
        
        project_root / "tests" / "data" / "sample_tracks.csv": '''event-id,visible,timestamp,location-long,location-lat,tag-local-identifier,individual-local-identifier
1,TRUE,2024-01-01 12:00:00,13.0,8.0,TEST001,1
2,TRUE,2024-01-01 13:00:00,13.01,8.01,TEST001,1
3,TRUE,2024-01-01 14:00:00,13.02,8.02,TEST001,1''',
        
        # Documentation files
        project_root / "docs" / "data_formats.md": '''# GPS Data Formats

## Expected CSV Format

Based on the provided elephant collar data, the system expects CSV files with the following columns:

### Required Columns:
- `timestamp`: Date and time of GPS fix (ISO format)
- `location-lat`: Latitude in decimal degrees
- `location-long`: Longitude in decimal degrees  
- `tag-local-identifier`: Collar identifier
- `individual-local-identifier`: Individual animal identifier

### Optional Columns:
- `event-id`: Unique event identifier
- `visible`: Data visibility flag
- `utm-easting`: UTM easting coordinate
- `utm-northing`: UTM northing coordinate
- `utm-zone`: UTM zone designation
- `study-name`: Name of the study
- `sensor-type`: Type of sensor used

## Supported Formats

The ingestion module supports:
- Standard CSV format with comma separation
- Various datetime formats (ISO 8601, custom formats)
- Both decimal degrees and UTM coordinates
- Multiple collar manufacturers' formats

## Quality Control

Data undergoes validation for:
- Coordinate bounds (lat: -90 to 90, lon: -180 to 180)
- Temporal ordering and gaps
- Movement speed filtering (>100 km/h flagged)
- Duplicate detection and removal
''',
        
        project_root / "docs" / "validation_rules.md": '''# GPS Data Validation Rules

## Coordinate Validation

### Latitude Bounds
- Valid range: -90.0 to 90.0 degrees
- Central Africa typical range: -5.0 to 15.0 degrees

### Longitude Bounds  
- Valid range: -180.0 to 180.0 degrees
- Central Africa typical range: 8.0 to 25.0 degrees

## Temporal Validation

### Timestamp Requirements
- Must be valid datetime format
- Should be in chronological order per individual
- Gaps > 24 hours flagged for review

### Sampling Intervals
- Typical collar intervals: 1-8 hours
- Sub-hourly intervals flagged as potential errors
- Intervals > 48 hours flagged as data gaps

## Movement Validation

### Speed Filtering
- Maximum realistic speed: 15 km/h sustained
- Burst speed threshold: 25 km/h
- Unrealistic speed threshold: 100 km/h (flagged/removed)

### Displacement Checks
- Maximum daily range: ~25 km
- Sudden large displacements (>50 km) flagged
- Zero displacement for >12 hours flagged

## Data Quality Flags

### Error Severity Levels
1. **Critical**: Invalid coordinates, corrupted timestamps
2. **Warning**: High speeds, large gaps, duplicates  
3. **Info**: Edge cases, unusual but valid patterns

### Handling Strategies
- **Critical errors**: Remove from dataset
- **Warnings**: Flag but retain with metadata
- **Info**: Log for quality reporting
''',
        
        # Notebook files
        project_root / "notebooks" / "01_data_exploration.ipynb": '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GPS Data Exploration\\n",
    "\\n",
    "Explore elephant GPS collar data structure and quality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import geopandas as gpd\\n",
    "import matplotlib.pyplot as plt\\n",
    "\\n",
    "# Placeholder for data exploration code"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}''',
        
        project_root / "notebooks" / "02_ingestion_examples.ipynb": '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GPS Data Ingestion Examples\\n",
    "\\n",
    "Examples of using the data ingestion module."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.data_ingestion import GPSDataProcessor\\n",
    "\\n",
    "# Example usage of GPS data processor"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}''',

        # Configuration files
        project_root / "data" / "raw" / ".gitkeep": "# Keep this directory in git",
        project_root / "data" / "processed" / ".gitkeep": "# Keep this directory in git", 
        project_root / "data" / "outputs" / ".gitkeep": "# Keep this directory in git",
        project_root / "logs" / ".gitkeep": "# Keep this directory in git"
    }
    
    print("Creating Step 2: Data Ingestion Module directory structure...")
    print("=" * 60)
    
    # Create directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory.relative_to(Path.cwd())}")
    
    print("\nCreating placeholder files...")
    print("-" * 40)
    
    # Create files with content
    for file_path, content in files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 Created file: {file_path.relative_to(Path.cwd())}")
    
    print("\n" + "=" * 60)
    print("✅ Step 2 directory structure created successfully!")
    print(f"📂 Project root: {project_root.absolute()}")
    
    # Display the structure
    print("\n📋 Directory Structure:")
    print_directory_tree(project_root)
    
    return project_root

def print_directory_tree(path, prefix="", is_last=True):
    """Recursively prints a directory tree structure"""
    if path.is_dir():
        print(f"{prefix}{'└── ' if is_last else '├── '}{path.name}/")
        children = sorted([p for p in path.iterdir() if not p.name.startswith('.')], 
                         key=lambda p: (p.is_file(), p.name.lower()))
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            extension = "    " if is_last else "│   "
            print_directory_tree(child, prefix + extension, is_last_child)
    else:
        icon = "📄" if path.suffix == ".py" else "📋" if path.suffix == ".md" else "📊" if path.suffix == ".ipynb" else "📝"
        print(f"{prefix}{'└── ' if is_last else '├── '}{icon} {path.name}")

if __name__ == "__main__":
    project_root = create_step2_structure()
    
    print("\n🚀 Next Steps:")
    print("1. Navigate to the elephant-corridors-project directory")
    print("2. We'll implement each file individually, starting with:")
    print("   - GPS validation functions")
    print("   - Data ingestion main module") 
    print("   - CRS utilities")
    print("   - Geometry utilities")
    print("3. Run unit tests to validate implementation")
    print("4. Test with real GPS collar data")
    
    print("\n📊 Based on your GPS data structure, the system will handle:")
    print("   - Movebank format CSV files")
    print("   - Columns: timestamp, location-lat, location-long, tag-local-identifier")
    print("   - UTM coordinates (optional)")
    print("   - Multiple collar files from Cameroon study sites")