"""
Elephant Corridor Analysis - Source Package
"""

# Import main classes - using absolute imports to avoid circular import issues
try:
    from data_ingestion import GPSDataProcessor
except ImportError:
    GPSDataProcessor = None

try:
    from utils.gps_validation import GPSValidator
except ImportError:
    GPSValidator = None

try:
    from utils.crs_utils import CRSUtils
except ImportError:
    CRSUtils = None

try:
    from utils.geometry_utils import GeometryUtils
except ImportError:
    GeometryUtils = None

try:
    from exceptions.validation_errors import (
        GPSValidationError, CoordinateValidationError, 
        TemporalValidationError, FileFormatError, CRSTransformationError
    )
except ImportError:
    # Define placeholder classes if imports fail
    class GPSValidationError(Exception):
        pass
    class CoordinateValidationError(GPSValidationError):
        pass
    class TemporalValidationError(GPSValidationError):
        pass
    class FileFormatError(GPSValidationError):
        pass
    class CRSTransformationError(GPSValidationError):
        pass

__version__ = "0.1.0"
__author__ = "Guillaume Atencia"

# Make main classes available at package level
__all__ = [
    'GPSDataProcessor',
    'GPSValidator', 
    'CRSUtils',
    'GeometryUtils',
    'GPSValidationError',
    'CoordinateValidationError',
    'TemporalValidationError', 
    'FileFormatError',
    'CRSTransformationError'
]