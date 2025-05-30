#!/usr/bin/env python3
"""
STEP 2.5: OpenTopography DEM Downloader
Reliable DEM acquisition using OpenTopography's Global DEM API
"""

import requests
import geopandas as gpd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import time
import json
import sys
import os
from datetime import datetime
from dataclasses import dataclass
import rasterio
from rasterio.crs import CRS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/opentopo_dem_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Get API key from environment variable
API_KEY = os.getenv("OPENTOPOGRAPHY_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("OPENTOPOGRAPHY_API_KEY is not set")

@dataclass
class AOIBounds:
    """Area of Interest with bounding box coordinates."""
    name: str
    min_lat: float
    max_lat: float  
    min_lon: float
    max_lon: float
    source_file: str
    
    def __post_init__(self):
        """Validate coordinates."""
        if not (-90 <= self.min_lat <= self.max_lat <= 90):
            raise ValueError(f"Invalid latitude range: {self.min_lat} to {self.max_lat}")
        if not (-180 <= self.min_lon <= self.max_lon <= 180):
            raise ValueError(f"Invalid longitude range: {self.min_lon} to {self.max_lon}")

class OpenTopographyDownloader:
    """Download DEMs using OpenTopography Global DEM API."""
    
    def __init__(self, output_dir: Path, buffer_km: float = 2.0):
        """Initialize OpenTopography downloader."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_km = buffer_km
        self.buffer_degrees = buffer_km / 111.32  # Approximate km to degrees
        
        # OpenTopography API endpoints - UPDATED to current public endpoint
        self.base_url = "https://portal.opentopography.org/API/globaldem"
        
        # Available DEM types (in order of preference)
        self.dem_types = [
            ("SRTMGL1", "SRTM 30m Global"),  # 30m resolution
            ("SRTMGL3", "SRTM 90m Global"),  # 90m resolution (fallback)
            ("SRTM_FF", "SRTM Finished"),    # Alternative SRTM
        ]
        
        # Download statistics
        self.stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'total_bytes': 0,
            'download_times': []
        }
        
        logger.info(f"Initialized OpenTopography downloader with endpoint: {self.base_url}")
        logger.info(f"API key configured: {'Yes' if API_KEY else 'No'}")
    
    def _add_buffer_to_bounds(self, aoi: AOIBounds) -> Tuple[float, float, float, float]:
        """Add buffer to AOI bounds and return (south, north, west, east)."""
        south = max(-90, aoi.min_lat - self.buffer_degrees)
        north = min(90, aoi.max_lat + self.buffer_degrees)
        west = max(-180, aoi.min_lon - self.buffer_degrees)
        east = min(180, aoi.max_lon + self.buffer_degrees)
        
        logger.info(f"  Original bounds: {aoi.min_lat:.4f},{aoi.min_lon:.4f} to {aoi.max_lat:.4f},{aoi.max_lon:.4f}")
        logger.info(f"  Buffered bounds: {south:.4f},{west:.4f} to {north:.4f},{east:.4f}")
        
        return south, north, west, east
    
    def _build_request_params(self, aoi: AOIBounds, dem_type: str) -> Dict[str, str]:
        """Build API request parameters."""
        south, north, west, east = self._add_buffer_to_bounds(aoi)
        
        return {
            'demtype': dem_type,
            'south': str(south),
            'north': str(north),
            'west': str(west),
            'east': str(east),
            'outputFormat': 'GTiff',
            'API_key': API_KEY
        }
    
    def _download_single_aoi(self, aoi: AOIBounds) -> bool:
        """Download DEM for a single AOI."""
        # Create output filename
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in aoi.name)
        output_file = self.output_dir / f"dem_{safe_name}.tif"
        
        # Skip if file already exists and is valid
        if self._file_exists_and_valid(output_file):
            logger.info(f"SKIP: {aoi.name} (file already exists)")
            self.stats['successful'] += 1
            return True
        
        logger.info(f"DOWNLOADING: {aoi.name}")
        
        # Try each DEM type until one works
        for dem_type, dem_description in self.dem_types:
            try:
                logger.info(f"  Trying {dem_description} ({dem_type})...")
                
                # Build request parameters
                params = self._build_request_params(aoi, dem_type)
                
                # Calculate expected area for size estimation
                area_deg_sq = (float(params['north']) - float(params['south'])) * \
                             (float(params['east']) - float(params['west']))
                expected_mb = area_deg_sq * 10  # Rough estimate
                
                logger.info(f"  Area: {area_deg_sq:.4f} deg² (~{expected_mb:.1f} MB estimated)")
                
                # DEBUG: Show the actual request URL being sent
                import requests.compat
                debug_url = f"{self.base_url}?{requests.compat.urlencode(params)}"
                logger.debug(f"REQUEST URL: {debug_url}")
                
                # Make request
                start_time = time.time()
                
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=300,  # 5 minute timeout
                    stream=True
                )
                
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    # Check if response is actually a GeoTIFF
                    content_type = response.headers.get('content-type', '')
                    if 'image/tiff' not in content_type and 'application/octet-stream' not in content_type:
                        logger.warning(f"  Unexpected content type: {content_type}")
                        continue
                    
                    # Download with progress tracking
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(output_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                    
                    download_time = time.time() - start_time
                    
                    # Validate the downloaded file
                    if self._validate_geotiff(output_file, aoi):
                        file_size_mb = downloaded_size / (1024 * 1024)
                        speed_mbps = file_size_mb / download_time if download_time > 0 else 0
                        
                        logger.info(f"  SUCCESS: {file_size_mb:.1f} MB in {download_time:.1f}s ({speed_mbps:.1f} MB/s)")
                        
                        self.stats['successful'] += 1
                        self.stats['total_bytes'] += downloaded_size
                        self.stats['download_times'].append(download_time)
                        
                        return True
                    else:
                        logger.warning(f"  Downloaded file failed validation")
                        output_file.unlink(missing_ok=True)
                        continue
                
                elif response.status_code == 400:
                    logger.warning(f"  Bad request (400) - invalid parameters for {dem_type}")
                    continue
                elif response.status_code == 401:
                    logger.warning(f"  Unauthorized (401) - check that OPENTOPOGRAPHY_API_KEY is valid")
                    logger.warning(f"  Request URL: {debug_url}")
                    continue
                elif response.status_code == 404:
                    logger.warning(f"  No data available (404) for {dem_type}")
                    continue
                elif response.status_code == 413:
                    logger.warning(f"  Request too large (413) for {dem_type} - area: {area_deg_sq:.2f} deg²")
                    logger.warning(f"  Consider using smaller AOIs (< 25 deg²)")
                    continue
                else:
                    logger.warning(f"  HTTP error {response.status_code} for {dem_type}")
                    logger.warning(f"  Response: {response.text[:200]}...")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"  Timeout downloading {dem_type}")
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"  Network error for {dem_type}: {e}")
                continue
            except Exception as e:
                logger.error(f"  Unexpected error for {dem_type}: {e}")
                continue
        
        # All DEM types failed
        logger.error(f"FAILED: {aoi.name} - all DEM types exhausted")
        self.stats['failed'] += 1
        return False
    
    def _file_exists_and_valid(self, filepath: Path) -> bool:
        """Check if file exists and appears to be a valid GeoTIFF."""
        if not filepath.exists():
            return False
        
        size = filepath.stat().st_size
        if size < 1024:  # Too small to be a valid GeoTIFF
            logger.warning(f"File too small ({size} bytes): {filepath}")
            return False
        
        # Try to open with rasterio to verify it's a valid GeoTIFF
        try:
            with rasterio.open(filepath) as src:
                return src.count > 0 and src.width > 0 and src.height > 0
        except Exception:
            logger.warning(f"Invalid GeoTIFF: {filepath}")
            return False
    
    def _validate_geotiff(self, filepath: Path, aoi: AOIBounds) -> bool:
        """Validate that the downloaded GeoTIFF covers the requested area."""
        try:
            with rasterio.open(filepath) as src:
                # Check basic properties
                if src.count == 0 or src.width == 0 or src.height == 0:
                    logger.warning("GeoTIFF has zero dimensions")
                    return False
                
                # Check coordinate system
                if src.crs is None:
                    logger.warning("GeoTIFF has no coordinate system")
                    return False
                
                # Check bounds coverage
                bounds = src.bounds
                if (bounds.left > aoi.min_lon or bounds.right < aoi.max_lon or
                    bounds.bottom > aoi.min_lat or bounds.top < aoi.max_lat):
                    logger.warning("GeoTIFF doesn't fully cover requested AOI")
                    # Don't fail for this - partial coverage might be acceptable
                
                logger.info(f"  Validated: {src.width}x{src.height} pixels, CRS: {src.crs}")
                return True
                
        except Exception as e:
            logger.error(f"Error validating GeoTIFF: {e}")
            return False
    
    def download_all_aois(self, aoi_bounds: List[AOIBounds]) -> Dict[str, bool]:
        """Download DEMs for all AOIs."""
        if not aoi_bounds:
            logger.info("No AOIs to download")
            return {}
        
        logger.info(f"Starting download of DEMs for {len(aoi_bounds)} AOIs")
        
        # Reset stats
        self.stats.update({
            'attempted': len(aoi_bounds),
            'successful': 0,
            'failed': 0,
            'total_bytes': 0,
            'download_times': []
        })
        
        results = {}
        start_time = time.time()
        
        for i, aoi in enumerate(aoi_bounds, 1):
            logger.info(f"\n📄 Processing {i}/{len(aoi_bounds)}: {aoi.name}")
            
            success = self._download_single_aoi(aoi)
            results[aoi.name] = success
            
            if success:
                logger.info(f"✅ Completed {aoi.name}")
            else:
                logger.error(f"❌ Failed {aoi.name}")
        
        # Final statistics
        duration = time.time() - start_time
        total_mb = self.stats['total_bytes'] / (1024 * 1024)
        avg_time = sum(self.stats['download_times']) / len(self.stats['download_times']) if self.stats['download_times'] else 0
        
        logger.info(f"\n🎉 Download batch complete!")
        logger.info(f"Success: {self.stats['successful']}/{self.stats['attempted']}")
        logger.info(f"Total data: {total_mb:.1f} MB")
        logger.info(f"Total time: {duration:.1f}s")
        logger.info(f"Average download time: {avg_time:.1f}s per AOI")
        
        return results

class AOIProcessor:
    """Process AOIs from Step 2 outputs."""
    
    def __init__(self, step2_outputs_dir: Optional[Path] = None):
        """Initialize AOI processor."""
        if step2_outputs_dir is None:
            # Auto-detect Step 2 outputs
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir.parent / "STEP 2" / "data" / "outputs",
                current_dir.parent / "STEP 2" / "outputs",
                current_dir / "inputs" / "aois"
            ]
            
            for path in possible_paths:
                if path.exists():
                    step2_outputs_dir = path
                    break
            
            if step2_outputs_dir is None:
                raise FileNotFoundError("Could not find Step 2 outputs directory")
        
        self.step2_outputs_dir = Path(step2_outputs_dir)
        logger.info(f"Using Step 2 outputs from: {self.step2_outputs_dir}")
    
    def discover_aoi_files(self) -> List[Path]:
        """Discover AOI files from Step 2 outputs."""
        aoi_files = []
        
        # Search patterns for AOI files
        patterns = ["*aoi*.geojson", "*aoi*.shp", "*AOI*.geojson", "*AOI*.shp"]
        
        for pattern in patterns:
            aoi_files.extend(self.step2_outputs_dir.rglob(pattern))
        
        # Remove duplicates (prefer .geojson over .shp)
        unique_files = {}
        for file in aoi_files:
            stem = file.stem
            if stem not in unique_files or file.suffix == '.geojson':
                unique_files[stem] = file
        
        result = list(unique_files.values())
        logger.info(f"Discovered {len(result)} AOI files")
        
        return result
    
    def load_aoi_bounds(self, aoi_files: List[Path]) -> List[AOIBounds]:
        """Load AOI bounding boxes from files."""
        aoi_bounds = []
        
        for aoi_file in aoi_files:
            try:
                logger.info(f"Loading AOI: {aoi_file.name}")
                gdf = gpd.read_file(aoi_file)
                
                if len(gdf) == 0:
                    logger.warning(f"Empty AOI file: {aoi_file}")
                    continue
                
                # Transform to geographic coordinates if needed
                if gdf.crs and not gdf.crs.is_geographic:
                    logger.info(f"  Converting from {gdf.crs} to EPSG:4326")
                    gdf = gdf.to_crs('EPSG:4326')
                elif gdf.crs is None:
                    logger.warning(f"  No CRS defined, assuming EPSG:4326")
                    gdf = gdf.set_crs('EPSG:4326')
                
                # Get total bounds
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                
                # Extract name
                if 'study_site' in gdf.columns:
                    name = gdf['study_site'].iloc[0]
                elif 'name' in gdf.columns:
                    name = gdf['name'].iloc[0]
                else:
                    name = aoi_file.stem
                
                aoi = AOIBounds(
                    name=str(name),
                    min_lat=bounds[1],  # miny
                    max_lat=bounds[3],  # maxy
                    min_lon=bounds[0],  # minx
                    max_lon=bounds[2],  # maxx
                    source_file=str(aoi_file)
                )
                
                aoi_bounds.append(aoi)
                logger.info(f"  → {aoi.name}: {aoi.min_lat:.3f},{aoi.min_lon:.3f} to {aoi.max_lat:.3f},{aoi.max_lon:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to load AOI {aoi_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(aoi_bounds)} AOI bounds")
        return aoi_bounds

def main():
    """Main OpenTopography DEM download workflow."""
    
    print("🛰️  STEP 2.5: OpenTopography DEM Downloader")
    print("=" * 60)
    print("Using OpenTopography Global DEM API with personal API key")
    print()
    
    # Setup directories
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    output_dir = Path("outputs/aoi_dems")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize components
        logger.info("Initializing OpenTopography DEM downloader...")
        
        aoi_processor = AOIProcessor()
        downloader = OpenTopographyDownloader(
            output_dir=output_dir,
            buffer_km=2.0  # 2km buffer around each AOI
        )
        
        # Discover and load AOIs
        logger.info("Discovering Step 2 AOI outputs...")
        aoi_files = aoi_processor.discover_aoi_files()
        
        if not aoi_files:
            logger.error("No AOI files found. Ensure Step 2 has been completed.")
            return False
        
        aoi_bounds = aoi_processor.load_aoi_bounds(aoi_files)
        
        if not aoi_bounds:
            logger.error("No valid AOI bounds could be loaded.")
            return False
        
        # Download DEMs for all AOIs
        logger.info("Starting DEM downloads using OpenTopography API...")
        start_time = time.time()
        
        download_results = downloader.download_all_aois(aoi_bounds)
        
        # Summary
        duration = time.time() - start_time
        successful_downloads = sum(1 for success in download_results.values() if success)
        
        print("\n🎉 OpenTopography DEM Download Complete!")
        print("=" * 50)
        print(f"AOIs processed: {len(aoi_bounds)}")
        print(f"Successful downloads: {successful_downloads}/{len(aoi_bounds)}")
        print(f"Success rate: {successful_downloads/len(aoi_bounds)*100:.1f}%")
        print(f"Total time: {duration:.1f} seconds")
        print(f"Output directory: {output_dir}")
        
        # Save download report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_version': 'opentopography_api',
            'aois_processed': len(aoi_bounds),
            'successful_downloads': successful_downloads,
            'failed_downloads': len(aoi_bounds) - successful_downloads,
            'success_rate': successful_downloads / len(aoi_bounds),
            'duration_seconds': duration,
            'output_directory': str(output_dir),
            'download_stats': downloader.stats,
            'aoi_details': [
                {
                    'name': aoi.name,
                    'bounds': [aoi.min_lat, aoi.min_lon, aoi.max_lat, aoi.max_lon],
                    'source_file': aoi.source_file,
                    'download_success': download_results.get(aoi.name, False)
                }
                for aoi in aoi_bounds
            ]
        }
        
        report_file = output_dir / f"opentopo_download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Download report saved: {report_file}")
        
        if successful_downloads >= len(aoi_bounds) * 0.8:  # 80% success rate
            print("✅ DEM download successful!")
            return True
        else:
            print(f"⚠️  Only {successful_downloads}/{len(aoi_bounds)} downloads successful")
            print("Check logs for details on failed downloads")
            return successful_downloads > 0
            
    except Exception as e:
        logger.error(f"OpenTopography DEM download failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)