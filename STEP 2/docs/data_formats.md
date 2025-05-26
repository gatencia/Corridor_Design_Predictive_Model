# GPS Data Formats

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
