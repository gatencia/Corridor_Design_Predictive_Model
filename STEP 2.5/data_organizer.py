#!/usr/bin/env python3
"""
Data Organization utilities for STEP 2.5
Organizes downloaded DEM data for Step 3 consumption.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import json
from datetime import datetime
import numpy as np
import rasterio
import rasterio.merge
import rasterio.mask
import rasterio.warp
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import box

logger = logging.getLogger(__name__)

class DataOrganizer:
    """
    Organizes downloaded DEM data for Step 3 consumption.
    
    Creates AOI-specific mosaics, maintains tile library, and generates
    metadata for efficient Step 3 processing.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize data organizer.
        
        Parameters:
        -----------
        output_dir : Path
            Base output directory for organized data
        """
        self.output_dir = Path(output_dir)
        
        # Create organized directory structure
        self.mosaics_dir = self.output_dir / "mosaics"
        self.tiles_dir = self.output_dir / "tiles"
        self.metadata_dir = self.output_dir / "metadata"
        self.temp_dir = self.output_dir / "temp"
        
        for directory in [self.mosaics_dir, self.tiles_dir, self.metadata_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data organizer initialized: {self.output_dir}")
    
    def extract_dem_tiles(self, download_results: List[Dict[str, Any]]) -> List[Path]:
        """
        Extract DEM tiles from zip files if necessary.
        
        Parameters:
        -----------
        download_results : List[Dict[str, Any]]
            List of download results from DEMDownloader
            
        Returns:
        --------
        List[Path]
            List of extracted/ready DEM file paths
        """
        extracted_files = []
        
        for result in download_results:
            file_path = Path(result['path'])
            
            if not file_path.exists():
                logger.warning(f"Downloaded file not found: {file_path}")
                continue
            
            if file_path.suffix.lower() == '.zip':
                # Extract zip file
                extract_dir = self.temp_dir / file_path.stem
                extract_dir.mkdir(exist_ok=True)
                
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Find DEM files in extracted directory
                    dem_extensions = ['.tif', '.tiff', '.hgt', '.bil', '.img']
                    for ext in dem_extensions:
                        dem_files = list(extract_dir.rglob(f'*{ext}'))
                        extracted_files.extend(dem_files)
                    
                    if not extracted_files:
                        # Look for any files that might be DEMs
                        all_files = list(extract_dir.rglob('*'))
                        for f in all_files:
                            if f.is_file() and f.suffix.lower() in dem_extensions:
                                extracted_files.append(f)
                    
                    logger.info(f"Extracted {len(extracted_files)} DEM files from {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to extract {file_path}: {e}")
            else:
                # File is already a DEM
                extracted_files.append(file_path)
        
        logger.info(f"Total DEM files ready: {len(extracted_files)}")
        return extracted_files
    
    def create_aoi_mosaic(self, dem_tiles: List[Path], 
                         aoi_bounds: Tuple[float, float, float, float],
                         aoi_name: str, buffer_km: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Create a mosaic DEM for a specific AOI.
        
        Parameters:
        -----------
        dem_tiles : List[Path]
            List of DEM tile file paths
        aoi_bounds : Tuple[float, float, float, float]
            AOI bounds (min_lon, min_lat, max_lon, max_lat)
        aoi_name : str
            Name for the AOI (used in filename)
        buffer_km : float
            Buffer around AOI in kilometers
            
        Returns:
        --------
        Optional[Dict[str, Any]]
            Mosaic information if successful, None if failed
        """
        logger.info(f"Creating mosaic for AOI: {aoi_name}")
        
        # Validate input tiles
        valid_tiles = []
        for tile_path in dem_tiles:
            if tile_path.exists():
                try:
                    # Quick validation - can we open the file?
                    with rasterio.open(tile_path) as src:
                        if src.width > 0 and src.height > 0:
                            valid_tiles.append(tile_path)
                        else:
                            logger.warning(f"Invalid tile dimensions: {tile_path}")
                except Exception as e:
                    logger.warning(f"Cannot read tile {tile_path}: {e}")
            else:
                logger.warning(f"Tile file not found: {tile_path}")
        
        if not valid_tiles:
            logger.error(f"No valid tiles found for AOI: {aoi_name}")
            return None
        
        logger.info(f"Using {len(valid_tiles)} valid tiles for mosaic")
        
        try:
            # Create output filename
            safe_name = "".join(c for c in aoi_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            output_filename = f"dem_mosaic_{safe_name}_{datetime.now().strftime('%Y%m%d')}.tif"
            output_path = self.mosaics_dir / output_filename
            
            # Open all valid tiles
            src_files_to_mosaic = []
            for tile_path in valid_tiles:
                src = rasterio.open(tile_path)
                src_files_to_mosaic.append(src)
            
            if len(src_files_to_mosaic) == 1:
                # Single tile - just clip to AOI
                src = src_files_to_mosaic[0]
                
                # Create AOI geometry with buffer
                buffered_bounds = self._buffer_bounds(aoi_bounds, buffer_km)
                aoi_geom = box(*buffered_bounds)
                
                # Clip to AOI
                clipped_array, clipped_transform = rasterio.mask.mask(
                    src, [aoi_geom], crop=True, nodata=src.nodata
                )
                
                # Update profile
                profile = src.profile.copy()
                profile.update({
                    'height': clipped_array.shape[1],
                    'width': clipped_array.shape[2],
                    'transform': clipped_transform,
                    'compress': 'lzw'
                })
                
                # Write mosaic
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(clipped_array)
                
            else:
                # Multiple tiles - create mosaic then clip
                mosaic_array, mosaic_transform = rasterio.merge.merge(
                    src_files_to_mosaic, nodata=src_files_to_mosaic[0].nodata
                )
                
                # Create temporary mosaic file
                temp_mosaic = self.temp_dir / f"temp_mosaic_{safe_name}.tif"
                
                profile = src_files_to_mosaic[0].profile.copy()
                profile.update({
                    'height': mosaic_array.shape[1],
                    'width': mosaic_array.shape[2],
                    'transform': mosaic_transform,
                    'compress': 'lzw'
                })
                
                with rasterio.open(temp_mosaic, 'w', **profile) as dst:
                    dst.write(mosaic_array)
                
                # Clip mosaic to AOI
                buffered_bounds = self._buffer_bounds(aoi_bounds, buffer_km)
                aoi_geom = box(*buffered_bounds)
                
                with rasterio.open(temp_mosaic) as mosaic_src:
                    clipped_array, clipped_transform = rasterio.mask.mask(
                        mosaic_src, [aoi_geom], crop=True, nodata=mosaic_src.nodata
                    )
                    
                    profile.update({
                        'height': clipped_array.shape[1],
                        'width': clipped_array.shape[2],
                        'transform': clipped_transform
                    })
                    
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(clipped_array)
                
                # Clean up temporary file
                if temp_mosaic.exists():
                    temp_mosaic.unlink()
            
            # Close all source files
            for src in src_files_to_mosaic:
                src.close()
            
            # Validate created mosaic
            mosaic_info = self._validate_mosaic(output_path, aoi_bounds, aoi_name)
            
            if mosaic_info:
                logger.info(f"Successfully created mosaic: {output_path}")
                return mosaic_info
            else:
                logger.error(f"Mosaic validation failed: {output_path}")
                if output_path.exists():
                    output_path.unlink()
                return None
                
        except Exception as e:
            logger.error(f"Failed to create mosaic for {aoi_name}: {e}")
            return None
    
    def _buffer_bounds(self, bounds: Tuple[float, float, float, float], 
                      buffer_km: float) -> Tuple[float, float, float, float]:
        """Add buffer to bounds in geographic coordinates."""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Convert km to degrees (approximate)
        lat_buffer = buffer_km / 111.0  # 1 degree ≈ 111 km
        lon_buffer = buffer_km / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))
        
        return (
            min_lon - lon_buffer,
            min_lat - lat_buffer, 
            max_lon + lon_buffer,
            max_lat + lat_buffer
        )
    
    def _validate_mosaic(self, mosaic_path: Path, 
                        expected_bounds: Tuple[float, float, float, float],
                        aoi_name: str) -> Optional[Dict[str, Any]]:
        """
        Validate created mosaic.
        
        Parameters:
        -----------
        mosaic_path : Path
            Path to mosaic file
        expected_bounds : Tuple[float, float, float, float]
            Expected bounds for validation
        aoi_name : str
            AOI name for metadata
            
        Returns:
        --------
        Optional[Dict[str, Any]]
            Mosaic info if valid, None if invalid
        """
        try:
            with rasterio.open(mosaic_path) as src:
                # Basic checks
                if src.width == 0 or src.height == 0:
                    logger.error(f"Mosaic has zero dimensions: {mosaic_path}")
                    return None
                
                if not src.crs:
                    logger.warning(f"Mosaic has no CRS: {mosaic_path}")
                
                # Read a sample to check for data
                sample_window = rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height))
                sample_data = src.read(1, window=sample_window)
                
                valid_pixels = np.sum(~np.isnan(sample_data) & (sample_data != src.nodata))
                if valid_pixels == 0:
                    logger.error(f"Mosaic contains no valid data: {mosaic_path}")
                    return None
                
                # Calculate statistics
                full_data = src.read(1, masked=True)
                if hasattr(full_data, 'compressed'):
                    valid_data = full_data.compressed()
                else:
                    valid_data = full_data[~np.isnan(full_data)]
                
                if len(valid_data) == 0:
                    logger.error(f"No valid elevation data in mosaic: {mosaic_path}")
                    return None
                
                stats = {
                    'min_elevation': float(np.min(valid_data)),
                    'max_elevation': float(np.max(valid_data)),
                    'mean_elevation': float(np.mean(valid_data)),
                    'std_elevation': float(np.std(valid_data))
                }
                
                # Check elevation range reasonableness
                elev_range = stats['max_elevation'] - stats['min_elevation']
                if elev_range < 1.0:
                    logger.warning(f"Very low elevation range: {elev_range:.2f}m")
                elif elev_range > 10000:
                    logger.warning(f"Very high elevation range: {elev_range:.2f}m")
                
                # File size check
                file_size_mb = mosaic_path.stat().st_size / (1024 * 1024)
                
                mosaic_info = {
                    'output_path': mosaic_path,
                    'aoi_name': aoi_name,
                    'bounds': src.bounds,
                    'crs': str(src.crs),
                    'shape': (src.height, src.width),
                    'resolution': (abs(src.transform[0]), abs(src.transform[4])),
                    'nodata_value': src.nodata,
                    'file_size_mb': file_size_mb,
                    'elevation_stats': stats,
                    'valid_pixels_percent': (len(valid_data) / (src.width * src.height)) * 100,
                    'created_timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Mosaic validation passed: {aoi_name}")
                logger.info(f"  Dimensions: {src.width}x{src.height}")
                logger.info(f"  Elevation range: {stats['min_elevation']:.1f} - {stats['max_elevation']:.1f}m")
                logger.info(f"  File size: {file_size_mb:.1f} MB")
                
                return mosaic_info
                
        except Exception as e:
            logger.error(f"Error validating mosaic {mosaic_path}: {e}")
            return None
    
    def organize_tile_library(self, download_results: List[Dict[str, Any]]) -> None:
        """
        Organize downloaded tiles into a structured library.
        
        Parameters:
        -----------
        download_results : List[Dict[str, Any]]
            Download results from DEMDownloader
        """
        logger.info("Organizing tile library...")
        
        for result in download_results:
            try:
                source_path = Path(result['path'])
                
                if not source_path.exists():
                    logger.warning(f"Source file not found: {source_path}")
                    continue
                
                # Determine source type from path
                if 'nasadem' in str(source_path).lower():
                    dest_dir = self.tiles_dir / "nasadem"
                elif 'srtm' in str(source_path).lower():
                    dest_dir = self.tiles_dir / "srtm"
                else:
                    dest_dir = self.tiles_dir / "other"
                
                dest_dir.mkdir(exist_ok=True)
                dest_path = dest_dir / source_path.name
                
                # Copy or move file
                if not dest_path.exists():
                    shutil.copy2(source_path, dest_path)
                    logger.debug(f"Copied tile to library: {dest_path.name}")
                
            except Exception as e:
                logger.error(f"Error organizing tile {result.get('tile_id', 'unknown')}: {e}")
        
        # Count organized tiles
        total_tiles = 0
        for source_dir in self.tiles_dir.iterdir():
            if source_dir.is_dir():
                tile_count = len(list(source_dir.glob("*")))
                total_tiles += tile_count
                logger.info(f"  {source_dir.name}: {tile_count} tiles")
        
        logger.info(f"Tile library organized: {total_tiles} total tiles")
    
    def create_metadata_summary(self, aoi_files: List[Dict[str, Any]],
                              download_results: Dict[str, Any],
                              organization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive metadata summary.
        
        Parameters:
        -----------
        aoi_files : List[Dict[str, Any]]
            Original AOI file information
        download_results : Dict[str, Any]
            Download results from DEMDownloader
        organization_results : Dict[str, Any]
            Organization results
            
        Returns:
        --------
        Dict[str, Any]
            Metadata summary information
        """
        logger.info("Creating metadata summary...")
        
        timestamp = datetime.now().isoformat()
        
        # Create comprehensive metadata
        metadata = {
            'processing_info': {
                'timestamp': timestamp,
                'step': '2.5',
                'description': 'Automated DEM download and organization for AOIs',
                'total_aois_processed': len(aoi_files),
                'total_tiles_downloaded': len(download_results.get('downloaded_files', [])),
                'total_mosaics_created': len(organization_results.get('mosaics_created', []))
            },
            'aoi_summary': {
                'total_aois': len(aoi_files),
                'aoi_details': aoi_files
            },
            'download_summary': {
                'download_time_seconds': download_results.get('download_time_seconds', 0),
                'total_size_mb': download_results.get('total_size_mb', 0),
                'successful_downloads': len(download_results.get('downloaded_files', [])),
                'failed_downloads': len(download_results.get('errors', [])),
                'error_summary': download_results.get('errors', [])
            },
            'organization_summary': {
                'mosaics_created': organization_results.get('mosaics_created', []),
                'organization_errors': organization_results.get('errors', [])
            },
            'directory_structure': {
                'base_directory': str(self.output_dir),
                'mosaics_directory': str(self.mosaics_dir),
                'tiles_directory': str(self.tiles_dir),
                'metadata_directory': str(self.metadata_dir)
            },
            'step3_readiness': {
                'ready_for_step3': len(organization_results.get('mosaics_created', [])) > 0,
                'dem_files_available': len(download_results.get('downloaded_files', [])),
                'processing_notes': [
                    "DEM mosaics are ready for EnergyScape processing",
                    "Use mosaics directory for Step 3 input",
                    "Individual tiles available in tiles directory for reference"
                ]
            }
        }
        
        # Save metadata
        metadata_file = self.metadata_dir / f"step25_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Create simple summary file for quick reference
        summary_file = self.metadata_dir / "processing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"STEP 2.5 Processing Summary\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"AOIs Processed: {len(aoi_files)}\n")
            f.write(f"DEM Tiles Downloaded: {len(download_results.get('downloaded_files', []))}\n")
            f.write(f"Mosaics Created: {len(organization_results.get('mosaics_created', []))}\n")
            f.write(f"Total Download Size: {download_results.get('total_size_mb', 0):.1f} MB\n")
            f.write(f"\nReady for Step 3: {'Yes' if metadata['step3_readiness']['ready_for_step3'] else 'No'}\n")
            
            if organization_results.get('mosaics_created'):
                f.write(f"\nCreated Mosaics:\n")
                for mosaic in organization_results['mosaics_created']:
                    f.write(f"  - {mosaic['aoi_name']}: {mosaic['output_path'].name}\n")
        
        logger.info(f"Metadata saved: {metadata_file}")
        logger.info(f"Summary saved: {summary_file}")
        
        return {
            'metadata_path': metadata_file,
            'summary_path': summary_file,
            'metadata': metadata
        }
    
    def cleanup_temporary_files(self) -> None:
        """Clean up temporary files and directories."""
        logger.info("Cleaning up temporary files...")
        
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temporary files: {e}")
    
    def get_organization_summary(self) -> Dict[str, Any]:
        """Get summary of organized data."""
        summary = {
            'base_directory': str(self.output_dir),
            'directories': {},
            'file_counts': {},
            'total_size_mb': 0.0
        }
        
        # Check each directory
        directories_to_check = [
            ('mosaics', self.mosaics_dir),
            ('tiles', self.tiles_dir),
            ('metadata', self.metadata_dir)
        ]
        
        for dir_name, dir_path in directories_to_check:
            if dir_path.exists():
                files = list(dir_path.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                summary['directories'][dir_name] = str(dir_path)
                summary['file_counts'][dir_name] = file_count
                summary['total_size_mb'] += size_mb
        
        return summary

# Utility functions
def organize_dem_data(download_results: Dict[str, Any], 
                     aoi_files: List[Dict[str, Any]],
                     output_dir: Path) -> Dict[str, Any]:
    """
    Simple function to organize DEM data.
    
    Parameters:
    -----------
    download_results : Dict[str, Any]
        Results from DEM download
    aoi_files : List[Dict[str, Any]]
        AOI file information
    output_dir : Path
        Output directory
        
    Returns:
    --------
    Dict[str, Any]
        Organization results
    """
    organizer = DataOrganizer(output_dir)
    
    # Extract tiles if needed
    dem_files = organizer.extract_dem_tiles(download_results['downloaded_files'])
    
    # Create mosaics
    mosaics_created = []
    for aoi in aoi_files:
        mosaic_info = organizer.create_aoi_mosaic(
            dem_tiles=dem_files,
            aoi_bounds=aoi['bounds'],
            aoi_name=aoi['study_site']
        )
        if mosaic_info:
            mosaics_created.append(mosaic_info)
    
    # Organize library
    organizer.organize_tile_library(download_results['downloaded_files'])
    
    # Create metadata
    organization_results = {
        'mosaics_created': mosaics_created,
        'errors': []
    }
    
    metadata = organizer.create_metadata_summary(
        aoi_files, download_results, organization_results
    )
    
    return organization_results

# Example usage and testing
if __name__ == "__main__":
    # Test data organizer
    print("📁 Testing Data Organizer")
    print("=" * 50)
    
    # Create test organizer
    test_output_dir = Path("test_organization")
    organizer = DataOrganizer(test_output_dir)
    
    # Test directory structure
    print("Created directory structure:")
    for dir_path in [organizer.mosaics_dir, organizer.tiles_dir, organizer.metadata_dir]:
        print(f"  ✅ {dir_path}")
    
    # Test bounds buffering
    test_bounds = (9.0, 4.0, 10.0, 5.0)
    buffered = organizer._buffer_bounds(test_bounds, 2.0)
    print(f"\nBuffer test:")
    print(f"  Original: {test_bounds}")
    print(f"  Buffered (2km): {buffered}")
    
    # Test organization summary
    summary = organizer.get_organization_summary()
    print(f"\nOrganization summary: {summary}")
    
    print("\n🎉 Data organizer working correctly!")