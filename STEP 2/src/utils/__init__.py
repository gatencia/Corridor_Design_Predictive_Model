"""
Utility modules for GPS data processing
"""

# Use try/except to handle import errors gracefully
try:
    from .gps_validation import GPSValidator
except ImportError as e:
    print(f"Warning: Could not import GPSValidator: {e}")
    GPSValidator = None

try:
    from .crs_utils import CRSUtils
except ImportError as e:
    print(f"Warning: Could not import CRSUtils: {e}")
    CRSUtils = None

try:
    from .geometry_utils import GeometryUtils
except ImportError as e:
    print(f"Warning: Could not import GeometryUtils: {e}")
    GeometryUtils = None

__all__ = ['GPSValidator', 'CRSUtils', 'GeometryUtils']