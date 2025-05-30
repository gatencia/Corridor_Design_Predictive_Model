#!/usr/bin/env python3
"""
STEP 2.5: Fixed DEM Acquisition System with NASA Earthdata Authentication
Fast, automated downloading of SRTM DEM tiles from NASA Earthdata for AOI-specific analysis.

FIXES:
1. Made SRTMTile hashable with frozen=True
2. Added coordinate transformation from projected to geographic coordinates
3. Fixed URL references and error handling
"""

import requests
import geopandas as gpd
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
import json
import sys

# Environment variable handling
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

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

@dataclass(frozen=True)  # FIX 1: Made hashable with frozen=True
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
    def zip_filename(self) -> str:
        """Generate ZIP filename for NASA downloads."""
        return f"{self.filename}.zip"
    
    @property
    def nasa_url(self) -> str:
        """Generate NASA Earthdata URL for SRTM tile."""
        # NASA SRTM GL1 (30m) - most reliable source
        base_url = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11"
        return f"{base_url}/{self.zip_filename}"
    
    @property 
    def fallback_urls(self) -> List[str]:
        """Alternative download URLs for this tile."""
        return [
            f"https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/{self.filename}",
            f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/{self.filename}",
            f"https://www.viewfinderpanoramas.org/dem3/{self.filename}",
        ]

class NASAEarthdataAuth:
    """Handle NASA Earthdata authentication."""
    
    def __init__(self):
        """Initialize NASA authentication."""
        self.username = None
        self.password = None
        self.session = None
        self._load_credentials()
        
    def _load_credentials(self):
        """Load NASA Earthdata credentials from environment or prompt."""
        # Try environment variables first
        self.username = os.getenv('NASA_EARTHDATA_USERNAME')
        self.password = os.getenv('NASA_EARTHDATA_PASSWORD')
        
        if not self.username or not self.password:
            logger.warning("NASA Earthdata credentials not found in environment variables")
            logger.info("Please set up your .env file with NASA_EARTHDATA_USERNAME and NASA_EARTHDATA_PASSWORD")
            
            # Fallback: use default credentials if provided
            if not self.username:
                logger.info("Using provided credentials for this session")
                self.username = "gatencia"  # Your username
                self.password = "wtfWTF12345!"  # Your password
            
        if self.username and self.password:
            logger.info(f"NASA Earthdata authentication configured for user: {self.username}")
        else:
            logger.error("No NASA Earthdata credentials available")
    
    def get_authenticated_session(self) -> requests.Session:
        """Get authenticated requests session for NASA downloads."""
        if self.session is None:
            self.session = requests.Session()
            self.session.auth = (self.username, self.password)
            
            # Set headers that NASA expects
            self.session.headers.update({
                'User-Agent': 'STEP25-DEM-Downloader/1.0'
            })
            
        return self.session
    
    def is_configured(self) -> bool:
        """Check if authentication is properly configured."""
        return bool(self.username and self.password)

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
    """High-speed SRTM tile downloader with NASA Earthdata authentication."""
    
    def __init__(self, output_dir: Path, max_workers: int = 6, 
                 max_retries: int = 3, timeout_seconds: int = 60):
        """
        Initialize SRTM downloader.
        
        Parameters:
        -----------
        output_dir : Path
            Directory to save downloaded tiles
        max_workers : int
            Maximum concurrent downloads (reduced for NASA servers)
        max_retries : int
            Maximum retry attempts per tile
        timeout_seconds : int
            HTTP request timeout (increased for larger NASA files)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        
        # Initialize NASA authentication
        self.nasa_auth = NASAEarthdataAuth()
        
        # Download statistics
        self.stats = {
            'attempted': 0,
            'successful': 0,
            'skipped': 0,
            'failed': 0,
            'total_bytes': 0,
            'nasa_downloads': 0,
            'fallback_downloads': 0
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
        
        # SRTM .hgt files should be around 2.9MB (1201x1201 int16)
        expected_size = 1201 * 1201 * 2  # ~2.9MB
        if size < expected_size * 0.3:  # Allow files to be 30% of expected (different formats)
            logger.warning(f"File suspiciously small ({size} bytes): {filepath}")
            return False
            
        return True
    
    def _extract_hgt_from_zip(self, zip_path: Path, target_path: Path) -> bool:
        """Extract .hgt file from NASA ZIP archive."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Find the .hgt file in the ZIP
                hgt_files = [f for f in zip_ref.namelist() if f.endswith('.hgt')]
                
                if not hgt_files:
                    logger.error(f"No .hgt file found in {zip_path}")
                    return False
                
                # Extract the first .hgt file
                hgt_file = hgt_files[0]
                
                # Extract to temporary location then move to target
                with zip_ref.open(hgt_file) as source:
                    with open(target_path, 'wb') as target:
                        target.write(source.read())
                
                logger.info(f"Extracted {hgt_file} from ZIP to {target_path.name}")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting ZIP {zip_path}: {e}")
            return False
    
    def _download_tile_with_retry(self, tile: SRTMTile) -> bool:
        """
        Download a single tile with retry logic and NASA authentication.
        
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
        
        # Try NASA first if authentication is available, then fallbacks
        urls_to_try = []
        
        if self.nasa_auth.is_configured():
            urls_to_try.append(('NASA', tile.nasa_url, True))  # (source, url, needs_auth)
        
        for url in tile.fallback_urls:
            urls_to_try.append(('Fallback', url, False))
        
        for attempt in range(self.max_retries):
            for source_name, url, needs_auth in urls_to_try:
                try:
                    logger.info(f"DOWNLOAD: {tile.filename} from {source_name} "
                               f"(attempt {attempt+1}/{self.max_retries})")
                    
                    # Choose session based on authentication needs
                    if needs_auth and self.nasa_auth.is_configured():
                        session = self.nasa_auth.get_authenticated_session()
                    else:
                        session = requests.Session()
                    
                    response = session.get(url, timeout=self.timeout, stream=True)
                    response.raise_for_status()
                    
                    # Determine if this is a ZIP file (NASA) or direct .hgt
                    is_zip = url.endswith('.zip') or 'zip' in response.headers.get('content-type', '')
                    
                    # Download to appropriate file
                    if is_zip:
                        temp_zip = self.output_dir / f"{tile.filename}.zip"
                        download_path = temp_zip
                    else:
                        download_path = filepath
                    
                    # Download with progress tracking
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(download_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                    
                    # Handle ZIP extraction if needed
                    if is_zip:
                        success = self._extract_hgt_from_zip(download_path, filepath)
                        download_path.unlink()  # Remove ZIP after extraction
                        
                        if not success:
                            continue  # Try next source
                    
                    # Verify final file
                    if self._file_exists_and_valid(filepath):
                        logger.info(f"SUCCESS: {tile.filename} from {source_name} "
                                   f"({downloaded_size:,} bytes)")
                        self.stats['successful'] += 1
                        self.stats['total_bytes'] += downloaded_size
                        
                        if needs_auth:
                            self.stats['nasa_downloads'] += 1
                        else:
                            self.stats['fallback_downloads'] += 1
                            
                        return True
                    else:
                        logger.error(f"INVALID: {tile.filename} failed validation")
                        filepath.unlink(missing_ok=True)
                        
                except requests.RequestException as e:
                    logger.warning(f"FAILED: {tile.filename} from {source_name}: {e}")
                    filepath.unlink(missing_ok=True)
                    
                    # Exponential backoff before retry
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) + (len(urls_to_try) * 0.5)
                        time.sleep(wait_time)
                        
                except Exception as e:
                    logger.error(f"ERROR: {tile.filename} from {source_name}: {e}")
                    filepath.unlink(missing_ok=True)
        
        logger.error(f"FAILED: {tile.filename} - all sources exhausted")
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
        if self.nasa_auth.is_configured():
            logger.info("NASA Earthdata authentication configured - using primary NASA sources")
        else:
            logger.warning("NASA Earthdata authentication not configured - using fallback sources only")
            
        start_time = time.time()
        results = {}
        
        # Reset stats for this batch
        self.stats.update({
            'attempted': len(tiles), 'successful': 0, 'skipped': 0, 'failed': 0, 
            'total_bytes': 0, 'nasa_downloads': 0, 'fallback_downloads': 0
        })
        
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
        logger.info(f"Sources: {self.stats['nasa_downloads']} NASA, {self.stats['fallback_downloads']} fallback")
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
        Load AOI bounding boxes from files with coordinate transformation.
        
        Parameters:
        -----------
        aoi_files : List[Path]
            List of AOI file paths
            
        Returns:
        --------
        List[AOIBounds]
            List of AOI bounding boxes in geographic coordinates
        """
        aoi_bounds = []
        
        for aoi_file in aoi_files:
            try:
                logger.info(f"Loading AOI: {aoi_file.name}")
                gdf = gpd.read_file(aoi_file)
                
                if len(gdf) == 0:
                    logger.warning(f"Empty AOI file: {aoi_file}")
                    continue
                
                # FIX 2: Transform to geographic coordinates if needed
                if gdf.crs and not gdf.crs.is_geographic:
                    logger.info(f"  Converting from {gdf.crs} to EPSG:4326")
                    gdf = gdf.to_crs('EPSG:4326')
                elif gdf.crs is None:
                    logger.warning(f"  No CRS defined, assuming EPSG:4326")
                    gdf = gdf.set_crs('EPSG:4326')
                
                # Get total bounds in geographic coordinates
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                
                # Validate that bounds are reasonable for geographic coordinates
                if not (-180 <= bounds[0] <= 180 and -180 <= bounds[2] <= 180):
                    logger.error(f"Invalid longitude bounds: {bounds[0]} to {bounds[2]}")
                    continue
                if not (-90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90):
                    logger.error(f"Invalid latitude bounds: {bounds[1]} to {bounds[3]}")
                    continue
                
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

def create_env_file():
    """Create .env file with NASA credentials."""
    env_file = Path(".env")
    
    if env_file.exists():
        logger.info(".env file already exists")
        return
    
    env_content = """# NASA Earthdata Credentials
# Get free account at: https://urs.earthdata.nasa.gov/users/new
NASA_EARTHDATA_USERNAME=gatencia
NASA_EARTHDATA_PASSWORD=wtfWTF12345!

# Optional: Add other configuration
SRTM_BUFFER_KM=2.0
MAX_CONCURRENT_DOWNLOADS=6
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    logger.info(f"Created .env file with NASA Earthdata credentials")
    logger.info("You can modify these credentials in the .env file if needed")

def main():
    """Main DEM acquisition workflow with NASA Earthdata."""
    
    print("🛰️  STEP 2.5: Fixed DEM Acquisition with NASA Earthdata")
    print("=" * 70)
    print("Objective: Fast download of SRTM tiles using NASA authentication")
    print("FIXES: Made SRTMTile hashable + coordinate transformation")
    print()
    
    # Setup directories
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    output_dir = Path("outputs/aoi_specific_dems")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create .env file if it doesn't exist
    create_env_file()
    
    try:
        # Initialize components
        logger.info("Initializing NASA Earthdata DEM acquisition system...")
        
        aoi_processor = AOIProcessor()
        tile_calculator = SRTMTileCalculator(buffer_km=2.0)
        downloader = SRTMDownloader(
            output_dir=output_dir,
            max_workers=6,  # Reduced for NASA server stability
            max_retries=3,
            timeout_seconds=60  # Increased for larger NASA files
        )
        
        # Verify NASA authentication
        if not downloader.nasa_auth.is_configured():
            logger.error("NASA Earthdata authentication not configured!")
            logger.error("Please check your .env file or environment variables")
            return False
        
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
        logger.info("Starting NASA Earthdata tile downloads...")
        start_time = time.time()
        
        download_results = downloader.download_tiles(all_tiles)
        
        # Summary
        duration = time.time() - start_time
        successful_tiles = sum(1 for success in download_results.values() if success)
        
        print("\n🎉 NASA Earthdata DEM Acquisition Complete!")
        print("=" * 50)
        print(f"AOIs processed: {len(aoi_bounds)}")
        print(f"Tiles downloaded: {successful_tiles}/{len(all_tiles)}")
        print(f"NASA downloads: {downloader.stats['nasa_downloads']}")
        print(f"Fallback downloads: {downloader.stats['fallback_downloads']}")
        print(f"Total time: {duration:.1f} seconds")
        print(f"Output directory: {output_dir}")
        
        # Save acquisition report
        report = {
            'timestamp': datetime.now().isoformat(),
            'nasa_earthdata_used': downloader.nasa_auth.is_configured(),
            'nasa_username': downloader.nasa_auth.username,
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
        
        report_file = output_dir / f"nasa_acquisition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"NASA Earthdata acquisition report saved: {report_file}")
        
        if successful_tiles == len(all_tiles):
            print("✅ All tiles downloaded successfully from NASA Earthdata!")
            return True
        else:
            failed_count = len(all_tiles) - successful_tiles
            print(f"⚠️  {failed_count} tiles failed to download")
            print("Check logs for details on failed downloads")
            return successful_tiles > 0  # Return True if at least some tiles succeeded
            
    except Exception as e:
        logger.error(f"NASA Earthdata DEM acquisition failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)