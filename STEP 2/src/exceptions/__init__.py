"""
Custom exception classes for GPS data processing
"""

try:
    from .validation_errors import (
        GPSValidationError, CoordinateValidationError, TemporalValidationError,
        FileFormatError, CRSTransformationError, DataQualityWarning,
        MovementValidationError, DuplicateDataError
    )
except ImportError as e:
    print(f"Warning: Could not import validation errors: {e}")
    
    # Define basic exception classes as fallback
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

    class DataQualityWarning(UserWarning):
        """Warning for data quality issues."""
        pass

    class MovementValidationError(GPSValidationError):
        """Exception for movement validation issues."""
        pass

    class DuplicateDataError(GPSValidationError):
        """Exception for duplicate data issues."""
        pass

__all__ = [
    'GPSValidationError', 
    'CoordinateValidationError', 
    'TemporalValidationError',
    'FileFormatError', 
    'CRSTransformationError', 
    'DataQualityWarning',
    'MovementValidationError', 
    'DuplicateDataError'
]