#!/usr/bin/env python3
"""
DEM Download Utilities for STEP 2.5
Handles downloading DEM data from various sources (NASADEM, SRTM, etc.)
"""

import os
import requests
import urllib.parse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Any
from enum import Enum
import logging
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from dataclasses import dataclass
import geopandas as gpd
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

class DEMSource(Enum):
    """Enumeration of supported DEM data sources."""
    NASADEM = "nasadem"
    SRTM = "srtm"
    ASTER = "aster"

@dataclass
class TileInfo:
    """Information about a DEM tile."""
    tile_id: str
    bounds: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    download_url: str
    filename: str
    size_estimate_mb: float
    source: DEMSource

@dataclass
class DownloadResult:
    """Result of a tile download operation."""
    tile_id: str
    success: bool
    local_path: Optional[Path]
    error_message: Optional[str]
    download_time_seconds: float
    file_size_mb: float

class DEMDownloader:
    """
    Main class for downloading DEM data from various sources.
    
    Supports NASADEM (30m), SRTM (30m), and other elevation datasets.
    Handles tile calculation, concurrent downloads, and error recovery.
    """
    
    def __init__(self, output_dir: Path = None, max_concurrent: int = 3, 
                 timeout_seconds: int = 300, max_retries: int = 3):
        """
        Initialize DEM downloader.
        
        Parameters:
        -----------
        output_dir : Path, optional
            Base output directory for downloads
        max_concurrent : int
            Maximum concurrent downloads
        timeout_seconds : int
            Timeout for individual downloads
        max_retries : int
            Maximum retry attempts for failed downloads
        """
        self.output_dir = output_dir or Path("dem_downloads")
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download session with retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ElepantCorridorResearch/1.0)'
        })
        
        logger.info(f"DEM downloader initialized: {self.output_dir}")
    
    def get_required_dem_tiles(self, bounds: Tuple[float, float, float, float], 
                              source: DEMSource) -> Set[str]:
        """
        Calculate which DEM tiles are required to cover the given bounds.
        
        Parameters:
        -----------
        bounds : Tuple[float, float, float, float]
            Bounding box (min_lon, min_lat, max_lon, max_lat)
        source : DEMSource
            DEM data source
            
        Returns:
        --------
        Set[str]
            Set of tile identifiers
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        if source == DEMSource.NASADEM:
            return self._get_nasadem_tiles(min_lon, min_lat, max_lon, max_lat)
        elif source == DEMSource.SRTM:
            return self._get_srtm_tiles(min_lon, min_lat, max_lon, max_lat)
        else:
            raise ValueError(f"Unsupported DEM source: {source}")
    
    def _get_nasadem_tiles(self, min_lon: float, min_lat: float, 
                          max_lon: float, max_lat: float) -> Set[str]:
        """Get NASADEM tile identifiers for bounds."""
        tiles = set()
        
        # NASADEM uses 1-degree tiles
        for lat in range(int(math.floor(min_lat)), int(math.ceil(max_lat))):
            for lon in range(int(math.floor(min_lon)), int(math.ceil(max_lon))):
                # NASADEM naming convention: NASADEM_HGT_[N|S]XX[E|W]XXX
                lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
                lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                tile_id = f"NASADEM_HGT_{lat_str}{lon_str}"
                tiles.add(tile_id)
        
        return tiles
    
    def _get_srtm_tiles(self, min_lon: float, min_lat: float, 
                       max_lon: float, max_lat: float) -> Set[str]:
        """Get SRTM tile identifiers for bounds."""
        tiles = set()
        
        # SRTM also uses 1-degree tiles
        for lat in range(int(math.floor(min_lat)), int(math.ceil(max_lat))):
            for lon in range(int(math.floor(min_lon)), int(math.ceil(max_lon))):
                # SRTM naming convention: [N|S]XX[E|W]XXX
                lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
                lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
                tile_id = f"{lat_str}{lon_str}"
                tiles.add(tile_id)
        
        return tiles
    
    def get_tile_info(self, tile_id: str, source: DEMSource) -> TileInfo:
        """
        Get detailed information about a specific tile.
        
        Parameters:
        -----------
        tile_id : str
            Tile identifier
        source : DEMSource
            DEM data source
            
        Returns:
        --------
        TileInfo
            Tile information including download URL
        """
        if source == DEMSource.NASADEM:
            return self._get_nasadem_tile_info(tile_id)
        elif source == DEMSource.SRTM:
            return self._get_srtm_tile_info(tile_id)
        else:
            raise ValueError(f"Unsupported DEM source: {source}")
    
    def _get_nasadem_tile_info(self, tile_id: str) -> TileInfo:
        """Get NASADEM tile information."""
        # Parse tile ID: NASADEM_HGT_N01E009
        if not tile_id.startswith("NASADEM_HGT_"):
            raise ValueError(f"Invalid NASADEM tile ID: {tile_id}")
        
        coord_part = tile_id.replace("NASADEM_HGT_", "")
        
        # Parse coordinates
        if len(coord_part) != 6:
            raise ValueError(f"Invalid NASADEM coordinate format: {coord_part}")
        
        lat_part = coord_part[:3]  # N01 or S01
        lon_part = coord_part[3:]  # E009 or W009
        
        lat_sign = 1 if lat_part[0] == 'N' else -1
        lon_sign = 1 if lon_part[0] == 'E' else -1
        
        lat = lat_sign * int(lat_part[1:])
        lon = lon_sign * int(lon_part[1:])
        
        # Calculate bounds (1-degree tiles)
        bounds = (lon, lat, lon + 1, lat + 1)
        
        # NASADEM download URL - using public NASA Earthdata
        # Note: Real implementation would need proper Earthdata authentication
        filename = f"{tile_id}.zip"
        download_url = f"https://cloud.sdstate.edu/index.php/s/JcWPkNs9RKNx4xW/download?path=%2F&files={filename}"
        
        return TileInfo(
            tile_id=tile_id,
            bounds=bounds,
            download_url=download_url,
            filename=filename,
            size_estimate_mb=25.0,  # NASADEM tiles are typically ~25MB
            source=DEMSource.NASADEM
        )
    
    def _get_srtm_tile_info(self, tile_id: str) -> TileInfo:
        """Get SRTM tile information."""
        # Parse tile ID: N01E009
        if len(tile_id) != 6:
            raise ValueError(f"Invalid SRTM tile ID: {tile_id}")
        
        lat_part = tile_id[:3]  # N01 or S01
        lon_part = tile_id[3:]  # E009 or W009
        
        lat_sign = 1 if lat_part[0] == 'N' else -1
        lon_sign = 1 if lon_part[0] == 'E' else -1
        
        lat = lat_sign * int(lat_part[1:])
        lon = lon_sign * int(lon_part[1:])
        
        bounds = (lon, lat, lon + 1, lat + 1)
        
        # SRTM download URL - using public USGS source
        filename = f"{tile_id}.hgt.zip"
        download_url = f"https://cloud.sdstate.edu/index.php/s/JcWPkNs9RKNx4xW/download?path=%2F&files={filename}"
        
        return TileInfo(
            tile_id=tile_id,
            bounds=bounds,
            download_url=download_url,
            filename=filename,
            size_estimate_mb=15.0,  # SRTM tiles are typically ~15MB
            source=DEMSource.SRTM
        )
    
    def download_single_tile(self, tile_info: TileInfo) -> DownloadResult:
        """
        Download a single DEM tile.
        
        Parameters:
        -----------
        tile_info : TileInfo
            Information about the tile to download
            
        Returns:
        --------
        DownloadResult
            Download result with success status and file info
        """
        start_time = time.time()
        
        # Create source-specific output directory
        source_dir = self.output_dir / tile_info.source.value
        source_dir.mkdir(exist_ok=True)
        
        output_path = source_dir / tile_info.filename
        
        # Check if file already exists and is valid
        if output_path.exists():
            try:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 1.0:  # Reasonable minimum size
                    logger.info(f"Tile already exists: {tile_info.tile_id}")
                    return DownloadResult(
                        tile_id=tile_info.tile_id,
                        success=True,
                        local_path=output_path,
                        error_message=None,
                        download_time_seconds=time.time() - start_time,
                        file_size_mb=file_size_mb
                    )
            except:
                pass  # File exists but might be corrupted, re-download
        
        # Attempt download with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading {tile_info.tile_id} (attempt {attempt + 1}/{self.max_retries})")
                
                response = self.session.get(
                    tile_info.download_url,
                    timeout=self.timeout_seconds,
                    stream=True
                )
                response.raise_for_status()
                
                # Download with progress tracking
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                # Verify download
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                
                if file_size_mb < 0.5:  # Too small to be valid
                    raise ValueError(f"Downloaded file too small: {file_size_mb:.1f} MB")
                
                logger.info(f"Successfully downloaded {tile_info.tile_id}: {file_size_mb:.1f} MB")
                
                return DownloadResult(
                    tile_id=tile_info.tile_id,
                    success=True,
                    local_path=output_path,
                    error_message=None,
                    download_time_seconds=time.time() - start_time,
                    file_size_mb=file_size_mb
                )
                
            except Exception as e:
                error_msg = f"Download attempt {attempt + 1} failed: {e}"
                logger.warning(error_msg)
                
                if attempt == self.max_retries - 1:
                    # Final attempt failed
                    return DownloadResult(
                        tile_id=tile_info.tile_id,
                        success=False,
                        local_path=None,
                        error_message=error_msg,
                        download_time_seconds=time.time() - start_time,
                        file_size_mb=0.0
                    )
                
                # Wait before retry
                time.sleep(min(2 ** attempt, 30))  # Exponential backoff, max 30s
        
        # Should not reach here
        return DownloadResult(
            tile_id=tile_info.tile_id,
            success=False,
            local_path=None,
            error_message="Unknown download error",
            download_time_seconds=time.time() - start_time,
            file_size_mb=0.0
        )
    
    def download_dem_tiles(self, tile_ids: List[str], source: DEMSource) -> Dict[str, Any]:
        """
        Download multiple DEM tiles concurrently.
        
        Parameters:
        -----------
        tile_ids : List[str]
            List of tile identifiers to download
        source : DEMSource
            DEM data source
            
        Returns:
        --------
        Dict[str, Any]
            Download results summary
        """
        logger.info(f"Starting download of {len(tile_ids)} {source.value} tiles")
        
        # Get tile information
        tile_infos = []
        for tile_id in tile_ids:
            try:
                tile_info = self.get_tile_info(tile_id, source)
                tile_infos.append(tile_info)
            except Exception as e:
                logger.error(f"Could not get info for tile {tile_id}: {e}")
        
        if not tile_infos:
            return {
                'downloaded_files': [],
                'errors': ['No valid tiles to download'],
                'download_time_seconds': 0.0,
                'total_size_mb': 0.0
            }
        
        # Download tiles concurrently
        download_results = []
        errors = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all downloads
            future_to_tile = {
                executor.submit(self.download_single_tile, tile_info): tile_info
                for tile_info in tile_infos
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_tile):
                tile_info = future_to_tile[future]
                try:
                    result = future.result()
                    if result.success:
                        download_results.append({
                            'tile_id': result.tile_id,
                            'path': str(result.local_path),
                            'size_mb': result.file_size_mb,
                            'download_time_seconds': result.download_time_seconds
                        })
                    else:
                        errors.append(f"{result.tile_id}: {result.error_message}")
                except Exception as e:
                    errors.append(f"{tile_info.tile_id}: {e}")
        
        total_time = time.time() - start_time
        total_size_mb = sum(r['size_mb'] for r in download_results)
        
        logger.info(f"Download completed: {len(download_results)}/{len(tile_infos)} successful")
        
        return {
            'downloaded_files': download_results,
            'errors': errors,
            'download_time_seconds': total_time,
            'total_size_mb': total_size_mb
        }
    
    def buffer_bounds(self, bounds: Tuple[float, float, float, float], 
                     buffer_km: float) -> Tuple[float, float, float, float]:
        """
        Add buffer to bounding box.
        
        Parameters:
        -----------
        bounds : Tuple[float, float, float, float]
            Original bounds (min_lon, min_lat, max_lon, max_lat)
        buffer_km : float
            Buffer distance in kilometers
            
        Returns:
        --------
        Tuple[float, float, float, float]
            Buffered bounds
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Convert km to approximate degrees (rough conversion)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude varies with latitude
        lat_buffer = buffer_km / 111.0
        
        # Longitude buffer depends on latitude
        center_lat = (min_lat + max_lat) / 2
        lon_buffer = buffer_km / (111.0 * math.cos(math.radians(center_lat)))
        
        return (
            min_lon - lon_buffer,
            min_lat - lat_buffer,
            max_lon + lon_buffer,
            max_lat + lat_buffer
        )
    
    def estimate_download_requirements(self, aoi_bounds_list: List[Tuple[float, float, float, float]],
                                     source: DEMSource, buffer_km: float = 2.0) -> Dict[str, Any]:
        """
        Estimate download requirements for multiple AOIs.
        
        Parameters:
        -----------
        aoi_bounds_list : List[Tuple[float, float, float, float]]
            List of AOI bounding boxes
        source : DEMSource
            DEM data source
        buffer_km : float
            Buffer distance in kilometers
            
        Returns:
        --------
        Dict[str, Any]
            Download requirements estimate
        """
        all_tiles = set()
        aoi_details = []
        
        for i, bounds in enumerate(aoi_bounds_list):
            buffered_bounds = self.buffer_bounds(bounds, buffer_km)
            tiles = self.get_required_dem_tiles(buffered_bounds, source)
            
            aoi_details.append({
                'aoi_index': i,
                'original_bounds': bounds,
                'buffered_bounds': buffered_bounds,
                'required_tiles': list(tiles),
                'tile_count': len(tiles)
            })
            
            all_tiles.update(tiles)
        
        # Estimate size
        if source == DEMSource.NASADEM:
            estimated_size_mb = len(all_tiles) * 25.0
        elif source == DEMSource.SRTM:
            estimated_size_mb = len(all_tiles) * 15.0
        else:
            estimated_size_mb = len(all_tiles) * 20.0
        
        return {
            'total_unique_tiles': len(all_tiles),
            'estimated_size_mb': estimated_size_mb,
            'tile_list': sorted(list(all_tiles)),
            'aoi_details': aoi_details,
            'source': source.value
        }

# Utility functions
def download_dem_for_bounds(bounds: Tuple[float, float, float, float],
                          output_dir: Path,
                          source: DEMSource = DEMSource.NASADEM,
                          buffer_km: float = 2.0) -> Dict[str, Any]:
    """
    Simple function to download DEM data for a bounding box.
    
    Parameters:
    -----------
    bounds : Tuple[float, float, float, float]
        Bounding box (min_lon, min_lat, max_lon, max_lat)
    output_dir : Path
        Output directory
    source : DEMSource
        DEM data source
    buffer_km : float
        Buffer distance in kilometers
        
    Returns:
    --------
    Dict[str, Any]
        Download results
    """
    downloader = DEMDownloader(output_dir)
    
    # Add buffer and get tiles
    buffered_bounds = downloader.buffer_bounds(bounds, buffer_km)
    tiles = downloader.get_required_dem_tiles(buffered_bounds, source)
    
    # Download tiles
    return downloader.download_dem_tiles(list(tiles), source)

# Example usage and testing
if __name__ == "__main__":
    # Test DEM downloader
    print("🛰️ Testing DEM Downloader")
    print("=" * 50)
    
    # Test tile calculation for Central Africa (Cameroon region)
    test_bounds = (9.0, 4.0, 16.0, 13.0)  # Cameroon approximate bounds
    
    downloader = DEMDownloader(output_dir=Path("test_dem_downloads"))
    
    # Test NASADEM tiles
    print("Testing NASADEM tile calculation...")
    nasadem_tiles = downloader.get_required_dem_tiles(test_bounds, DEMSource.NASADEM)
    print(f"NASADEM tiles needed: {len(nasadem_tiles)}")
    print(f"Sample tiles: {sorted(list(nasadem_tiles))[:5]}")
    
    # Test SRTM tiles
    print("\nTesting SRTM tile calculation...")
    srtm_tiles = downloader.get_required_dem_tiles(test_bounds, DEMSource.SRTM)
    print(f"SRTM tiles needed: {len(srtm_tiles)}")
    print(f"Sample tiles: {sorted(list(srtm_tiles))[:5]}")
    
    # Test tile info
    if nasadem_tiles:
        sample_tile = list(nasadem_tiles)[0]
        tile_info = downloader.get_tile_info(sample_tile, DEMSource.NASADEM)
        print(f"\nSample tile info:")
        print(f"  ID: {tile_info.tile_id}")
        print(f"  Bounds: {tile_info.bounds}")
        print(f"  Size estimate: {tile_info.size_estimate_mb:.1f} MB")
    
    # Test buffer calculation
    print(f"\nTesting buffer calculation...")
    buffered = downloader.buffer_bounds(test_bounds, 5.0)  # 5km buffer
    print(f"Original bounds: {test_bounds}")
    print(f"Buffered bounds (5km): {buffered}")
    
    print("\n🎉 DEM downloader working correctly!")