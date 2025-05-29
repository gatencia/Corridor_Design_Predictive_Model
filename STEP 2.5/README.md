# STEP 2.5: Automated DEM Download for AOIs

**Bridges Step 2 (GPS Processing) → Step 3 (EnergyScape) with targeted DEM data acquisition**

## Overview

STEP 2.5 automatically downloads Digital Elevation Model (DEM) data for the Areas of Interest (AOIs) generated in Step 2, preparing the elevation data needed for energy landscape calculations in Step 3.

### Why STEP 2.5?

- **🎯 Targeted**: Downloads DEM data only for your actual study areas
- **🤖 Automated**: No manual tile selection or download management needed  
- **🔗 Integrated**: Seamlessly bridges Step 2 outputs with Step 3 requirements
- **📦 Organized**: Creates ready-to-use mosaics and maintains tile libraries
- **⚡ Efficient**: Concurrent downloads with retry logic and progress tracking

## Features

### 🛰️ **DEM Data Sources**
- **NASADEM** (30m resolution) - Primary choice for elephant studies
- **SRTM** (30m resolution) - Alternative global coverage
- **ASTER GDEM** (30m resolution) - Future support planned

### 📐 **Smart AOI Processing**
- Discovers AOI files from Step 2 outputs automatically
- Calculates required DEM tiles with configurable buffer zones
- Handles multiple file formats (GeoJSON, Shapefile)
- Removes duplicates and validates AOI data

### 📥 **Robust Downloading**
- Concurrent downloads with configurable limits
- Automatic retry logic with exponential backoff
- Progress tracking and size estimation
- Resume capability for interrupted downloads

### 📁 **Data Organization**
- Creates AOI-specific DEM mosaics ready for Step 3
- Maintains organized tile library for reference
- Generates comprehensive metadata for tracking
- Validates data quality and Step 3 readiness

## Installation

### Prerequisites

Ensure you have completed **Step 2** (GPS data processing) before running Step 2.5.

### Install Dependencies

```bash
# Navigate to STEP 2.5 directory
cd "STEP 2.5"

# Install additional dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Quick dependency check
python -c "import requests, geopy, rasterio; print('✅ STEP 2.5 dependencies ready')"
```

## Quick Start

### Basic Usage

```bash
# Run with default settings
python download_dem_for_aois.py

# Use 5km buffer around AOIs
python download_dem_for_aois.py --buffer 5.0

# Specify output directory
python download_dem_for_aois.py --output-dir /path/to/dem/storage
```

### Dry Run (Recommended First)

```bash
# See what would be downloaded without downloading
python download_dem_for_aois.py --dry-run
```

Example output:
```
🔍 DRY RUN - Would download:
   📦 12 tiles
   💾 ~300 MB
   📍 Tiles: ['NASADEM_HGT_N04E009', 'NASADEM_HGT_N04E010', ...]
```

## Command Line Options

```bash
python download_dem_for_aois.py [OPTIONS]

Options:
  --output-dir PATH        Output directory for DEM data 
                          (default: ../STEP 3/data/raw/dem)
  --buffer FLOAT          Buffer around AOIs in kilometers (default: 2.0)
  --source CHOICE         DEM source: nasadem|srtm (default: nasadem)
  --max-concurrent INT    Max concurrent downloads (default: 3)
  --dry-run              Show what would be downloaded
  --log-level CHOICE     Logging level: DEBUG|INFO|WARNING|ERROR
  --help                 Show help message
```

## Examples

### Standard Workflow

```bash
# 1. Check what will be downloaded
python download_dem_for_aois.py --dry-run

# 2. Download with 3km buffer
python download_dem_for_aois.py --buffer 3.0

# 3. Verify results
ls -la "../STEP 3/data/raw/dem/mosaics/"
```

### Custom Configuration

```bash
# Large buffer for regional analysis
python download_dem_for_aois.py --buffer 10.0 --max-concurrent 5

# Use SRTM instead of NASADEM
python download_dem_for_aois.py --source srtm

# Store in custom location
python download_dem_for_aois.py --output-dir /data/elephant_dems/
```

### Troubleshooting Mode

```bash
# Debug mode with detailed logging
python download_dem_for_aois.py --log-level DEBUG
```

## Output Structure

STEP 2.5 creates an organized directory structure ready for Step 3:

```
output_directory/
├── mosaics/                    # Ready-to-use AOI mosaics
│   ├── dem_mosaic_Site1_20241201.tif
│   ├── dem_mosaic_Site2_20241201.tif
│   └── ...
├── tiles/                      # Tile library
│   ├── nasadem/
│   │   ├── NASADEM_HGT_N04E009.zip
│   │   └── ...
│   └── srtm/
│       └── ...
├── metadata/                   # Processing metadata
│   ├── step25_metadata_20241201_143022.json
│   └── processing_summary.txt
└── temp/                      # Temporary files (auto-cleaned)
```

## Integration with Project Workflow

### Prerequisites: Step 2 Complete
```bash
# Ensure Step 2 has generated AOI files
ls "../STEP 2/data/outputs/individual_aois/"
```

### STEP 2.5: DEM Download
```bash
python download_dem_for_aois.py
```

### Next: Step 3 Ready
```bash
# Verify Step 3 readiness
ls "../STEP 3/data/raw/dem/mosaics/"

# Proceed to Step 3
cd "../STEP 3"
python run_energyscape.py
```

## Supported DEM Sources

### NASADEM (Recommended)
- **Resolution**: 30 meters
- **Coverage**: Global (60°N to 56°S)
- **Quality**: High accuracy, void-filled
- **Use Case**: Primary choice for elephant corridor analysis
- **Tile Size**: ~25 MB per 1° × 1° tile

### SRTM
- **Resolution**: 30 meters (SRTM Plus)
- **Coverage**: Global (60°N to 56°S) 
- **Quality**: Good, some voids in steep terrain
- **Use Case**: Alternative when NASADEM unavailable
- **Tile Size**: ~15 MB per 1° × 1° tile

## Configuration

### Buffer Zones

The buffer parameter controls how much area around each AOI is included:

- **1-2 km**: Minimal buffer, exact AOI coverage
- **3-5 km**: Recommended for most analyses
- **5-10 km**: Large buffer for regional context
- **>10 km**: Full landscape analysis

### Download Settings

Adjust based on your internet connection and server load:

```bash
# Conservative (slow connection)
--max-concurrent 2

# Standard (good connection)
--max-concurrent 3

# Aggressive (fast connection)
--max-concurrent 5
```

## Troubleshooting

### Common Issues

#### No AOI Files Found
```
❌ No AOI files found from Step 2
```
**Solution**: Ensure Step 2 processing completed successfully
```bash
# Check for AOI files
find "../STEP 2" -name "*aoi*" -o -name "*AOI*"

# Re-run Step 2 if needed
cd "../STEP 2"
python run_gps_processing.py
```

#### Download Failures
```
❌ Failed to download tile NASADEM_HGT_N04E009
```
**Solutions**:
- Check internet connection
- Try again later (server might be busy)
- Use alternative source: `--source srtm`
- Reduce concurrent downloads: `--max-concurrent 2`

#### Large Download Size
```
📦 Estimated download: 2.5 GB
```
**Solutions**:
- Reduce buffer: `--buffer 1.0`
- Process AOIs individually
- Check disk space

### Validation Errors

#### Invalid Mosaic
```
❌ Mosaic validation failed
```
**Solutions**:
- Check source DEM data quality
- Verify AOI geometry validity
- Try different DEM source

### Performance Issues

#### Slow Downloads
- Reduce `--max-concurrent` 
- Check network bandwidth
- Try different time of day

#### High Memory Usage
- Process fewer AOIs at once
- Use smaller buffer zones
- Close other applications

## Advanced Usage

### Processing Specific AOIs

```python
# Python script for custom processing
from aoi_processor import AOIProcessor
from dem_downloader import DEMDownloader

# Discover AOIs
processor = AOIProcessor(project_root=".")
aois = processor.find_step2_aoi_outputs()

# Filter by area
large_aois = [aoi for aoi in aois if aoi['area_km2'] > 100]

# Download for specific AOIs
downloader = DEMDownloader(output_dir="custom_dem")
# ... custom processing
```

### Custom DEM Sources

Extend the downloader for additional sources:

```python
# Add to dem_downloader.py
class DEMSource(Enum):
    NASADEM = "nasadem"
    SRTM = "srtm"
    CUSTOM = "custom"  # Your addition
```

## Data Quality Checks

STEP 2.5 automatically validates:

- ✅ **File integrity**: Downloaded files are complete
- ✅ **Spatial coverage**: Mosaics cover AOI bounds
- ✅ **Elevation range**: Reasonable elevation values
- ✅ **Data completeness**: Minimal voids/gaps
- ✅ **Step 3 readiness**: Compatible format and structure

## Monitoring and Logs

### Log Files
```bash
# View current processing
tail -f logs/step25_dem_download_*.log

# Check processing summary
cat "output_dir/metadata/processing_summary.txt"
```

### Progress Tracking
- Real-time download progress
- Estimated completion time
- Data validation status
- Error reporting with solutions

## Support and Troubleshooting

### Quick Diagnostics
```bash
# Check system readiness
python -c "
import requests, rasterio, geopandas
print('✅ Core dependencies ready')
"

# Verify Step 2 outputs
python -c "
from aoi_processor import AOIProcessor
p = AOIProcessor('.')
aois = p.find_step2_aoi_outputs()
print(f'✅ Found {len(aois)} AOIs from Step 2')
"
```

### Getting Help

1. **Check logs**: Look in `logs/` directory for detailed error messages
2. **Run with debug**: `--log-level DEBUG` for verbose output
3. **Validate inputs**: Ensure Step 2 completed successfully
4. **Check disk space**: DEM downloads can be large
5. **Test connectivity**: Verify internet access to data servers

### Common Solutions

| Issue | Solution |
|-------|----------|
| No AOIs found | Complete Step 2 first |
| Download fails | Check internet, try `--max-concurrent 2` |
| Large file size | Reduce `--buffer` parameter |
| Out of disk space | Clean temp files, use external drive |
| Validation fails | Try different DEM source |

## Performance Benchmarks

Typical performance on a good internet connection:

- **AOI Discovery**: < 30 seconds
- **Tile Calculation**: < 5 seconds  
- **Download (per tile)**: 30-60 seconds
- **Mosaic Creation**: 1-5 minutes per AOI
- **Total Time**: 15-45 minutes for 5-10 AOIs

## Version History

- **v1.0**: Initial implementation with NASADEM support
- **v1.1**: Added SRTM support and improved error handling
- **v1.2**: Enhanced validation and Step 3 integration

---

**Next Step**: Proceed to [Step 3: EnergyScape Implementation](../STEP%203/README.md) to generate energy landscape surfaces using your downloaded DEM data.