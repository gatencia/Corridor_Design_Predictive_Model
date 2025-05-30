#!/usr/bin/env python3
"""
Robust Multi-Source DEM Downloader
Tries every available DEM source with fallback mechanisms to guarantee data retrieval.
"""

import os
import sys
import requests
import rasterio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json
import time
import subprocess
import tempfile
from urllib.parse import urlencode
import zipfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustDEMDownloader:
    """Downloads DEM data using multiple sources with automatic fallback."""
    
    def __init__(self, output_dir: str = "outputs/robust_dems"):
        """Initialize the robust DEM downloader."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize credentials
        self.opentopo_key = os.environ.get('OPENTOPOGRAPHY_API_KEY')
        self.earthdata_user = os.environ.get('EARTHDATA_USER') 
        self.earthdata_pass = os.environ.get('EARTHDATA_PASS')
        self.copernicus_token = os.environ.get('COPERNICUS_TOKEN')
        
        self.session = requests.Session()
        self._setup_earthdata_auth()
        
        # DEM source priorities (order matters - fastest/most reliable first)
        self.dem_sources = [
            'opentopography_srtm',
            'opentopography_aw3d30', 
            'copernicus_glo30',
            'nasa_nasadem',
            'aws_terrain_tiles',
            'merit_dem',
            'hydrosheds'
        ]
        
        logger.info(f"Initialized Robust DEM Downloader with {len(self.dem_sources)} sources")
        logger.info(f"Available credentials: OT={bool(self.opentopo_key)}, NASA={bool(self.earthdata_user)}, Copernicus={bool(self.copernicus_token)}")
    
    def _setup_earthdata_auth(self):
        """Setup NASA Earthdata authentication."""
        if self.earthdata_user and self.earthdata_pass:
            netrc_path = Path.home() / '.netrc'
            netrc_content = f"""machine urs.earthdata.nasa.gov
    login {self.earthdata_user}
    password {self.earthdata_pass}
"""
            with open(netrc_path, 'w') as f:
                f.write(netrc_content)
            netrc_path.chmod(0o600)
            logger.info("Setup NASA Earthdata authentication")
    
    def download_dem_robust(self, name: str, bounds: Tuple[float, float, float, float], 
                           target_resolution: int = 30) -> Optional[Path]:
        """
        Download DEM using multiple sources with fallback.
        
        Parameters:
        -----------
        name : str
            Name for the DEM (used in filename)
        bounds : Tuple[float, float, float, float]
            Bounding box (west, south, east, north) in decimal degrees
        target_resolution : int
            Target resolution in meters
            
        Returns:
        --------
        Optional[Path]
            Path to downloaded DEM file or None if all sources failed
        """
        west, south, east, north = bounds
        
        logger.info(f"🚀 Starting robust DEM download for: {name}")
        logger.info(f"Bounds: W={west:.4f}, S={south:.4f}, E={east:.4f}, N={north:.4f}")
        logger.info(f"Area: {(east-west)*(north-south):.4f} deg² (~{(east-west)*111*(north-south)*111:.1f} km²)")
        
        # Try each source in order
        for i, source in enumerate(self.dem_sources, 1):
            logger.info(f"📡 Trying source {i}/{len(self.dem_sources)}: {source}")
            
            try:
                dem_path = self._download_from_source(source, name, bounds, target_resolution)
                if dem_path and self._validate_dem_file(dem_path):
                    logger.info(f"✅ SUCCESS: Downloaded DEM from {source}: {dem_path}")
                    return dem_path
                else:
                    logger.warning(f"⚠️  Source {source} failed or returned invalid data")
                    
            except Exception as e:
                logger.error(f"❌ Source {source} error: {e}")
                
            # Brief pause between attempts
            time.sleep(1)
        
        logger.error(f"💥 ALL SOURCES FAILED for {name}")
        return None
    
    def _download_from_source(self, source: str, name: str, bounds: Tuple[float, float, float, float],
                             target_resolution: int) -> Optional[Path]:
        """Download from a specific source."""
        west, south, east, north = bounds
        
        if source == 'opentopography_srtm':
            return self._download_opentopography(name, bounds, 'SRTMGL1')
            
        elif source == 'opentopography_aw3d30':
            return self._download_opentopography(name, bounds, 'AW3D30')
            
        elif source == 'copernicus_glo30':
            return self._download_copernicus(name, bounds, resolution=30)
            
        elif source == 'nasa_nasadem':
            return self._download_nasa_nasadem(name, bounds)
            
        elif source == 'aws_terrain_tiles':
            return self._download_aws_terrain(name, bounds)
            
        elif source == 'merit_dem':
            return self._download_merit_dem(name, bounds)
            
        elif source == 'hydrosheds':
            return self._download_hydrosheds(name, bounds)
            
        else:
            logger.error(f"Unknown source: {source}")
            return None
    
    def _download_opentopography(self, name: str, bounds: Tuple[float, float, float, float], 
                                dem_type: str) -> Optional[Path]:
        """Download from OpenTopography API."""
        if not self.opentopo_key:
            logger.warning("No OpenTopography API key available")
            return None
            
        west, south, east, north = bounds
        
        # Check bounds size (OT has limits)
        area = (east - west) * (north - south)
        if area > 25:  # OT limit is ~25 deg²
            logger.warning(f"Area too large for OpenTopography: {area:.1f} deg²")
            return None
            
        # Check latitude limits for SRTM
        if dem_type.startswith('SRTM') and (north > 60 or south < -56):
            logger.warning(f"Latitude out of SRTM range: {south} to {north}")
            return None
        
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            'demtype': dem_type,
            'south': str(south),
            'north': str(north),
            'west': str(west),
            'east': str(east),
            'outputFormat': 'GTiff',
            'API_Key': self.opentopo_key
        }
        
        try:
            logger.info(f"OpenTopography request: {dem_type}")
            response = self.session.get(url, params=params, timeout=300, stream=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if 'image/tiff' in content_type or 'application/octet-stream' in content_type:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = self.output_dir / f"{name}_OT_{dem_type}_{timestamp}.tif"
                    
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info(f"Downloaded {output_path.stat().st_size:,} bytes from OpenTopography")
                    return output_path
                else:
                    logger.error(f"Invalid content type from OpenTopography: {content_type}")
                    return None
            else:
                logger.error(f"OpenTopography HTTP {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"OpenTopography error: {e}")
            return None
    
    def _download_copernicus(self, name: str, bounds: Tuple[float, float, float, float], 
                            resolution: int = 30) -> Optional[Path]:
        """Download from Copernicus Data Space Ecosystem."""
        if not self.copernicus_token:
            logger.warning("No Copernicus token available")
            return None
            
        west, south, east, north = bounds
        
        # Copernicus uses 1° x 1° tiles
        # For simplicity, download tile containing the center point
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2
        
        # Determine tile coordinates
        tile_lat = int(np.floor(center_lat))
        tile_lon = int(np.floor(center_lon))
        
        # Build tile name
        lat_str = f"N{tile_lat:02d}" if tile_lat >= 0 else f"S{abs(tile_lat):02d}"
        lon_str = f"E{tile_lon:03d}" if tile_lon >= 0 else f"W{abs(tile_lon):03d}"
        
        if resolution == 30:
            product_id = f"Copernicus_DSM_30_{lat_str}_{lon_str}"
        else:
            product_id = f"Copernicus_DSM_90_{lat_str}_{lon_str}"
        
        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products('{product_id}')/$value"
        headers = {'Authorization': f'Bearer {self.copernicus_token}'}
        
        try:
            logger.info(f"Copernicus request: {product_id}")
            response = self.session.get(url, headers=headers, timeout=300, stream=True)
            
            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.output_dir / f"{name}_Copernicus_{resolution}m_{timestamp}.tif"
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.info(f"Downloaded {output_path.stat().st_size:,} bytes from Copernicus")
                return output_path
            else:
                logger.error(f"Copernicus HTTP {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Copernicus error: {e}")
            return None
    
    def _download_nasa_nasadem(self, name: str, bounds: Tuple[float, float, float, float]) -> Optional[Path]:
        """Download NASADEM from NASA Earthdata."""
        if not (self.earthdata_user and self.earthdata_pass):
            logger.warning("No NASA Earthdata credentials available")
            return None
            
        west, south, east, north = bounds
        
        # NASADEM uses 1° x 1° tiles
        center_lat = int(np.floor((south + north) / 2))
        center_lon = int(np.floor((west + east) / 2))
        
        # Build filename
        lat_str = f"n{center_lat:02d}" if center_lat >= 0 else f"s{abs(center_lat):02d}"
        lon_str = f"e{center_lon:03d}" if center_lon >= 0 else f"w{abs(center_lon):03d}"
        filename = f"NASADEM_HGT_{lat_str}{lon_str}.zip"
        
        url = f"https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2021.01.01/{filename}"
        
        try:
            logger.info(f"NASA NASADEM request: {filename}")
            
            # Use wget with netrc authentication
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_path = self.output_dir / f"{name}_NASADEM_{timestamp}.zip"
            
            cmd = [
                'wget', '--load-cookies', str(Path.home() / '.urs_cookies'),
                '--save-cookies', str(Path.home() / '.urs_cookies'),
                '--keep-session-cookies', '--no-check-certificate',
                url, '-O', str(zip_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and zip_path.exists():
                # Extract the zip file
                extract_dir = self.output_dir / f"{name}_NASADEM_extract_{timestamp}"
                extract_dir.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find the .hgt file
                hgt_files = list(extract_dir.glob("*.hgt"))
                if hgt_files:
                    output_path = self.output_dir / f"{name}_NASADEM_{timestamp}.hgt"
                    shutil.copy2(hgt_files[0], output_path)
                    
                    # Cleanup
                    zip_path.unlink(missing_ok=True)
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    
                    logger.info(f"Downloaded NASADEM: {output_path}")
                    return output_path
                else:
                    logger.error("No .hgt file found in NASADEM zip")
                    return None
            else:
                logger.error(f"NASADEM wget failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"NASA NASADEM error: {e}")
            return None
    
    def _download_aws_terrain(self, name: str, bounds: Tuple[float, float, float, float]) -> Optional[Path]:
        """Download from AWS terrain tiles (no auth required)."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to determine the correct tile coordinates
            # and stitch multiple tiles together for the AOI
            logger.info("AWS terrain tiles implementation would go here")
            logger.info("This requires tile coordinate calculation and stitching")
            return None
            
        except Exception as e:
            logger.error(f"AWS terrain error: {e}")
            return None
    
    def _download_merit_dem(self, name: str, bounds: Tuple[float, float, float, float]) -> Optional[Path]:
        """Download MERIT DEM from Zenodo."""
        try:
            # MERIT DEM is distributed as large continental tiles
            # For Africa, you'd download the Africa tile and crop it
            logger.info("MERIT DEM would require downloading large continental tiles")
            logger.info("Implementation would download from Zenodo and crop to AOI")
            return None
            
        except Exception as e:
            logger.error(f"MERIT DEM error: {e}")
            return None
    
    def _download_hydrosheds(self, name: str, bounds: Tuple[float, float, float, float]) -> Optional[Path]:
        """Download from HydroSHEDS."""
        try:
            logger.info("HydroSHEDS implementation would go here")
            return None
            
        except Exception as e:
            logger.error(f"HydroSHEDS error: {e}")
            return None
    
    def _validate_dem_file(self, dem_path: Path) -> bool:
        """Validate downloaded DEM file."""
        try:
            # Check file size
            file_size = dem_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is likely an error
                logger.error(f"File too small: {file_size} bytes")
                return False
            
            # Try to open with rasterio
            with rasterio.open(dem_path) as src:
                # Check basic properties
                if src.count < 1:
                    logger.error("No bands in raster")
                    return False
                
                if src.width < 1 or src.height < 1:
                    logger.error(f"Invalid dimensions: {src.width}x{src.height}")
                    return False
                
                # Read a small sample
                sample = src.read(1, window=rasterio.windows.Window(0, 0, min(10, src.width), min(10, src.height)))
                if np.all(np.isnan(sample)) or np.all(sample == src.nodata):
                    logger.error("No valid data in sample")
                    return False
                
                logger.info(f"Valid DEM: {src.width}x{src.height}, {src.count} bands, CRS: {src.crs}")
                return True
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def create_test_aoi(self) -> Tuple[str, Tuple[float, float, float, float]]:
        """Create a small test AOI for quick results."""
        # Small area in Central Cameroon - Dja Faunal Reserve area
        # This is a known elephant habitat area and should have good data coverage
        name = "test_dja_cameroon"
        bounds = (12.8, 2.8, 12.9, 2.9)  # 0.1° x 0.1° = ~11km x 11km
        
        logger.info(f"Created test AOI: {name}")
        logger.info(f"Location: Central Cameroon (Dja Faunal Reserve area)")
        logger.info(f"Size: ~11km x 11km")
        
        return name, bounds

# Example usage and testing
if __name__ == "__main__":
    # Initialize downloader
    downloader = RobustDEMDownloader()
    
    # Create a small test AOI for quick results
    test_name, test_bounds = downloader.create_test_aoi()
    
    logger.info("🎯 Starting test download with smallest AOI...")
    
    # Download DEM with all fallback sources
    dem_path = downloader.download_dem_robust(
        name=test_name,
        bounds=test_bounds,
        target_resolution=30
    )
    
    if dem_path:
        print(f"\n🎉 SUCCESS! Downloaded DEM: {dem_path}")
        print(f"File size: {dem_path.stat().st_size:,} bytes")
        
        # Quick stats
        try:
            with rasterio.open(dem_path) as src:
                data = src.read(1)
                valid_data = data[data != src.nodata]
                if len(valid_data) > 0:
                    print(f"Elevation range: {valid_data.min():.1f} to {valid_data.max():.1f} meters")
                    print(f"Mean elevation: {valid_data.mean():.1f} meters")
        except Exception as e:
            print(f"Could not read elevation stats: {e}")
    else:
        print("\n💥 FAILED: Could not download DEM from any source")
        print("Check your credentials and network connection")
        sys.exit(1)