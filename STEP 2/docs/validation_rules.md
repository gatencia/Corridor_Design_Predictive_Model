# GPS Data Validation Rules

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
