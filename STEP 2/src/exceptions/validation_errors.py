"""
Custom exception classes for data validation errors.
"""

class GPSValidationError(Exception):
    """Base exception for GPS data validation errors."""
    pass

class CoordinateValidationError(GPSValidationError):
    """Exception for invalid coordinate values."""
    def __init__(self, message: str, invalid_coordinates=None):
        super().__init__(message)
        self.invalid_coordinates = invalid_coordinates

class TemporalValidationError(GPSValidationError):
    """Exception for temporal data issues."""
    def __init__(self, message: str, problematic_timestamps=None):
        super().__init__(message)
        self.problematic_timestamps = problematic_timestamps

class FileFormatError(GPSValidationError):
    """Exception for file format issues."""
    def __init__(self, message: str, missing_columns=None):
        super().__init__(message)
        self.missing_columns = missing_columns

class CRSTransformationError(GPSValidationError):
    """Exception for CRS transformation issues."""
    def __init__(self, message: str, source_crs=None, target_crs=None):
        super().__init__(message)
        self.source_crs = source_crs
        self.target_crs = target_crs

class DataQualityWarning(UserWarning):
    """Warning for data quality issues that don't prevent processing."""
    pass

class MovementValidationError(GPSValidationError):
    """Exception for movement validation issues."""
    def __init__(self, message: str, high_speed_fixes=None):
        super().__init__(message)
        self.high_speed_fixes = high_speed_fixes

class DuplicateDataError(GPSValidationError):
    """Exception for duplicate data issues."""
    def __init__(self, message: str, duplicate_count=None):
        super().__init__(message)
        self.duplicate_count = duplicate_count