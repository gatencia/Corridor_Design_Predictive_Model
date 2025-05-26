"""
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
