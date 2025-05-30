

#!/usr/bin/env python3
"""
STEP 2.5: Streamlined DEM Acquisition System
Fast, automated downloading of SRTM DEM tiles from AWS S3 for AOI-specific analysis.

Focus: Pure acquisition speed and reliability. No processing, just raw tile downloads.
"""

import requests
import geopandas as gpd
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
from dataclasses import dataclass
from datetime import datetime
import json
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dem_acquisition.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

@dataclass  
class SRTMTile:
    """SRTM tile identifier and metadata."""
    lat: int  # Integer degree latitude (south edge)
    lon: int  # Integer degree longitude (west edge)
    
    @property
    def filename(self) -> str:
        """Generate standard SRTM filename (e.g., N01E009.hgt)."""
        lat_str = f"{'N' if self.lat >= 0 else 'S'}{abs(self.lat):02d}"
        lon_str = f"{'E' if self.lon >= 0 else 'W'}{abs(self.lon):03d}"
        return f"{lat_str}{lon_str}.hgt"
    
    @property
    def s3_url(self) -> str:
    """Generate AWS S3 URL for SRTM tile."""
        # OLD (broken): base_url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1"
        # NEW (working):
        base_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF"
        return f"{base_url}/{self.filename}"
    
    @property 
    def fallback_urls(self) -> List[str]:
        """Alternative download URLs for this tile."""
        return [
            f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/{self._get_skadi_path()}",
            f"https://www.viewfinderpanoramas.org/dem3/{self.filename}",
        ]
    
    def _get_continent(self) -> str:
        """Determine continent directory for USGS structure."""
        if 15 <= self.lat <= 83 and -180 <= self.lon <= 180:
            return "Eurasia"
        elif -60 <= self.lat <= 15 and -180 <= self.lon <= -30:
            return "North_America" if self.lat >= 15 else "South_America"
        elif -60 <= self.lat <= 37 and -20 <= self.lon <= 180:
            return "Africa"
        else:
            return "Australia"

class SRTMTileCalculator:
    """Calculate required SRTM tiles for given AOIs."""
    
    def __init__(self, buffer_km: float = 2.0):
        """
        Initialize tile calculator.
        
        Parameters:
        -----------
        buffer_km : float
            Buffer distance in kilometers to add around AOIs
        """
        self.buffer_km = buffer_km
        self.buffer_degrees = buffer_km / 111.32  # Approximate km to degrees conversion
        
    def calculate_tiles_for_aoi(self, aoi: AOIBounds) -> Set[SRTMTile]:
        """
        Calculate SRTM tiles needed for an AOI with buffer.
        
        Parameters:
        -----------
        aoi : AOIBounds
            Area of Interest boundaries
            
        Returns:
        --------
        Set[SRTMTile]
            Set of required SRTM tiles
        """
        # Add buffer to AOI bounds
        buffered_min_lat = aoi.min_lat - self.buffer_degrees
        buffered_max_lat = aoi.max_lat + self.buffer_degrees
        buffered_min_lon = aoi.min_lon - self.buffer_degrees
        buffered_max_lon = aoi.max_lon + self.buffer_degrees
        
        # Calculate tile bounds (SRTM uses 1-degree tiles)
        min_tile_lat = math.floor(buffered_min_lat)
        max_tile_lat = math.floor(buffered_max_lat)
        min_tile_lon = math.floor(buffered_min_lon)
        max_tile_lon = math.floor(buffered_max_lon)
        
        # Generate tile set
        tiles = set()
        for lat in range(min_tile_lat, max_tile_lat + 1):
            for lon in range(min_tile_lon, max_tile_lon + 1):
                tiles.add(SRTMTile(lat, lon))
        
        logger.info(f"AOI '{aoi.name}': {len(tiles)} tiles required "
                   f"(bounds: {buffered_min_lat:.3f},{buffered_min_lon:.3f} to "
                   f"{buffered_max_lat:.3f},{buffered_max_lon:.3f})")
        
        return tiles

class SRTMDownloader:
    """High-speed SRTM tile downloader with S3 optimization."""
    
    def __init__(self, output_dir: Path, max_workers: int = 8, 
                 max_retries: int = 3, timeout_seconds: int = 30):
        """
        Initialize SRTM downloader.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save downloaded tiles
        max_workers : int
            Maximum concurrent downloads
        max_retries : int
            Maximum retry attempts per tile
        timeout_seconds : int
            HTTP request timeout
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        
        # Download statistics
        self.stats = {
            'attempted': 0,
            'successful': 0,
            'skipped': 0,
            'failed': 0,
            'total_bytes': 0
        }
    
    def _file_exists_and_valid(self, filepath: Path) -> bool:
        """Check if file exists and appears valid."""
        if not filepath.exists():
            return False
        
        # Basic validation: file size > 0 and reasonable for SRTM tile
        size = filepath.stat().st_size
        if size == 0:
            logger.warning(f"Removing zero-size file: {filepath}")
            filepath.unlink()
            return False
        
        # SRTM .hgt files should be around 25MB (1201x1201 int16)
        expected_size = 1201 * 1201 * 2  # ~2.9MB
        if size < expected_size * 0.5:  # Allow 50% smaller (compressed/different format)
            logger.warning(f"File suspiciously small ({size} bytes): {filepath}")
            return False
            
        return True
    
    def _download_tile_with_retry(self, tile: SRTMTile) -> bool:
        """
        Download a single tile with retry logic.
        
        Parameters:
        -----------
        tile : SRTMTile
            Tile to download
            
        Returns:
        --------
        bool
            True if download successful
        """
        filepath = self.output_dir / tile.filename
        
        # Skip if file already exists and is valid
        if self._file_exists_and_valid(filepath):
            logger.info(f"SKIP: {tile.filename} (already exists)")
            self.stats['skipped'] += 1
            return True
        
        # Try primary S3 URL first, then fallbacks
        urls_to_try = [tile.s3_url] + tile.fallback_urls
        
        for attempt in range(self.max_retries):
            for url_idx, url in enumerate(urls_to_try):
                try:
                    logger.info(f"DOWNLOAD: {tile.filename} (attempt {attempt+1}/{self.max_retries}, "
                               f"source {url_idx+1}/{len(urls_to_try)})")
                    
                    response = requests.get(url, timeout=self.timeout, stream=True)
                    response.raise_for_status()
                    
                    # Download with progress tracking
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                    
                    # Verify download
                    if self._file_exists_and_valid(filepath):
                        logger.info(f"SUCCESS: {tile.filename} ({downloaded_size:,} bytes)")
                        self.stats['successful'] += 1
                        self.stats['total_bytes'] += downloaded_size
                        return True
                    else:
                        logger.error(f"INVALID: {tile.filename} failed validation")
                        filepath.unlink(missing_ok=True)
                        
                except requests.RequestException as e:
                    logger.warning(f"FAILED: {tile.filename} from {url}: {e}")
                    filepath.unlink(missing_ok=True)
                    
                    # Exponential backoff before retry
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + (url_idx * 0.5)
                        time.sleep(wait_time)
                        
                except Exception as e:
                    logger.error(f"ERROR: {tile.filename}: {e}")
                    filepath.unlink(missing_ok=True)
        
        logger.error(f"FAILED: {tile.filename} - all attempts exhausted")
        self.stats['failed'] += 1
        return False
    
    def download_tiles(self, tiles: Set[SRTMTile]) -> Dict[str, bool]:
        """
        Download multiple tiles concurrently.
        
        Parameters:
        -----------
        tiles : Set[SRTMTile]
            Set of tiles to download
            
        Returns:
        --------
        Dict[str, bool]
            Mapping of tile filename to success status
        """
        if not tiles:
            logger.info("No tiles to download")
            return {}
        
        logger.info(f"Starting download of {len(tiles)} tiles using {self.max_workers} workers")
        start_time = time.time()
        results = {}
        
        # Reset stats for this batch
        self.stats.update({'attempted': len(tiles), 'successful': 0, 'skipped': 0, 'failed': 0, 'total_bytes': 0})
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_tile = {
                executor.submit(self._download_tile_with_retry, tile): tile 
                for tile in tiles
            }
            
            # Process completed downloads
            for future in as_completed(future_to_tile):
                tile = future_to_tile[future]
                try:
                    success = future.result()
                    results[tile.filename] = success
                except Exception as e:
                    logger.error(f"Exception downloading {tile.filename}: {e}")
                    results[tile.filename] = False
                    self.stats['failed'] += 1
        
        # Log final statistics
        duration = time.time() - start_time
        successful = self.stats['successful']
        total_mb = self.stats['total_bytes'] / (1024 * 1024)
        speed_mbps = total_mb / duration if duration > 0 else 0
        
        logger.info(f"BATCH COMPLETE: {successful}/{len(tiles)} successful in {duration:.1f}s")
        logger.info(f"Downloaded {total_mb:.1f} MB at {speed_mbps:.2f} MB/s")
        logger.info(f"Stats: {self.stats['successful']} success, {self.stats['skipped']} skipped, {self.stats['failed']} failed")
        
        return results

class AOIProcessor:
    """Process AOIs from Step 2 outputs and coordinate DEM acquisition."""
    
    def __init__(self, step2_outputs_dir: Optional[Path] = None):
        """
        Initialize AOI processor.
        
        Parameters:
        -----------
        step2_outputs_dir : Path, optional
            Directory containing Step 2 AOI outputs
        """
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
        
        # Remove duplicates (keep .geojson over .shp)
        unique_files = {}
        for file in aoi_files:
            stem = file.stem
            if stem not in unique_files or file.suffix == '.geojson':
                unique_files[stem] = file
        
        result = list(unique_files.values())
        logger.info(f"Discovered {len(result)} AOI files")
        
        return result
    
    def load_aoi_bounds(self, aoi_files: List[Path]) -> List[AOIBounds]:
        """
        Load AOI bounding boxes from files.
        
        Parameters:
        -----------
        aoi_files : List[Path]
            List of AOI file paths
            
        Returns:
        --------
        List[AOIBounds]
            List of AOI bounding boxes
        """
        aoi_bounds = []
        
        for aoi_file in aoi_files:
            try:
                logger.info(f"Loading AOI: {aoi_file.name}")
                gdf = gpd.read_file(aoi_file)
                
                if len(gdf) == 0:
                    logger.warning(f"Empty AOI file: {aoi_file}")
                    continue
                
                # Get total bounds
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                
                # Extract name from file or attributes
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
    """Main DEM acquisition workflow."""
    
    print("🏔️  STEP 2.5: Streamlined DEM Acquisition")
    print("=" * 60)
    print("Objective: Fast download of SRTM tiles for all Step 2 AOIs")
    print()
    
    # Setup directories
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    output_dir = Path("outputs/aoi_specific_dems")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize components
        logger.info("Initializing DEM acquisition system...")
        
        aoi_processor = AOIProcessor()
        tile_calculator = SRTMTileCalculator(buffer_km=2.0)
        downloader = SRTMDownloader(
            output_dir=output_dir,
            max_workers=8,  # Aggressive concurrent downloads
            max_retries=3,
            timeout_seconds=30
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
        
        # Calculate all required tiles
        logger.info("Calculating required SRTM tiles...")
        all_tiles = set()
        aoi_tile_map = {}
        
        for aoi in aoi_bounds:
            tiles = tile_calculator.calculate_tiles_for_aoi(aoi)
            all_tiles.update(tiles)
            aoi_tile_map[aoi.name] = tiles
        
        logger.info(f"Total unique tiles required: {len(all_tiles)}")
        
        # Download all tiles
        logger.info("Starting concurrent tile downloads...")
        start_time = time.time()
        
        download_results = downloader.download_tiles(all_tiles)
        
        # Summary
        duration = time.time() - start_time
        successful_tiles = sum(1 for success in download_results.values() if success)
        
        print("\n🎉 DEM Acquisition Complete!")
        print("=" * 40)
        print(f"AOIs processed: {len(aoi_bounds)}")
        print(f"Tiles downloaded: {successful_tiles}/{len(all_tiles)}")
        print(f"Total time: {duration:.1f} seconds")
        print(f"Output directory: {output_dir}")
        
        # Save acquisition report
        report = {
            'timestamp': datetime.now().isoformat(),
            'aois_processed': len(aoi_bounds),
            'total_tiles_required': len(all_tiles),
            'successful_downloads': successful_tiles,
            'failed_downloads': len(all_tiles) - successful_tiles,
            'duration_seconds': duration,
            'output_directory': str(output_dir),
            'download_stats': downloader.stats,
            'aoi_details': [
                {
                    'name': aoi.name,
                    'bounds': [aoi.min_lat, aoi.min_lon, aoi.max_lat, aoi.max_lon],
                    'tiles_required': len(aoi_tile_map[aoi.name]),
                    'source_file': aoi.source_file
                }
                for aoi in aoi_bounds
            ],
            'tile_details': [
                {
                    'filename': tile.filename,
                    'lat': tile.lat,
                    'lon': tile.lon,
                    'success': download_results.get(tile.filename, False)
                }
                for tile in sorted(all_tiles, key=lambda t: (t.lat, t.lon))
            ]
        }
        
        report_file = output_dir / f"acquisition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Acquisition report saved: {report_file}")
        
        if successful_tiles == len(all_tiles):
            print("✅ All tiles downloaded successfully!")
            return True
        else:
            failed_count = len(all_tiles) - successful_tiles
            print(f"⚠️  {failed_count} tiles failed to download")
            print("Check logs for details on failed downloads")
            return False
            
    except Exception as e:
        logger.error(f"DEM acquisition failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)