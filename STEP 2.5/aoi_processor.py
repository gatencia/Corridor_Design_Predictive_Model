#!/usr/bin/env python3
"""
AOI Processing Utilities for STEP 2.5
Processes AOI polygons from Step 2 and determines DEM tile requirements.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from shapely.geometry import box
import math

logger = logging.getLogger(__name__)

class AOIProcessor:
    """Process AOI polygons and determine DEM requirements."""
    
    def __init__(self):
        """Initialize AOI processor."""
        self.processed_aois = []
        
    def process_aoi_file(self, aoi_file_path: str, aoi_gdf: gpd.GeoDataFrame,
                        buffer_km: float = 2.0) -> List[Dict[str, Any]]:
        """
        Process a single AOI file and extract information.
        
        Parameters:
        -----------
        aoi_file_path : str
            Path to AOI file
        aoi_gdf : gpd.GeoDataFrame
            Loaded AOI GeoDataFrame
        buffer_km : float
            Buffer distance in kilometers
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of processed AOI information
        """
        aoi_file_path = Path(aoi_file_path)
        processed_aois = []
        
        # Ensure data is in geographic coordinates (WGS84)
        if aoi_gdf.crs != 'EPSG:4326':
            logger.info(f"Reprojecting AOI from {aoi_gdf.crs} to WGS84")
            aoi_gdf = aoi_gdf.to_crs('EPSG:4326')
        
        for idx, row in aoi_gdf.iterrows():
            try:
                geometry = row.geometry
                
                # Get basic information
                aoi_info = {
                    'source_file': str(aoi_file_path),
                    'aoi_index': idx,
                    'geometry': geometry,
                    'original_bounds': geometry.bounds,  # (minx, miny, maxx, maxy)
                    'area_km2': getattr(row, 'area_km2', 0),
                    'study_site': getattr(row, 'study_site', aoi_file_path.stem),
                    'utm_crs': getattr(row, 'utm_crs', None)
                }
                
                # Add buffer to bounds (convert km to degrees, roughly)
                # 1 degree ≈ 111 km, so buffer in degrees = buffer_km / 111
                buffer_deg = buffer_km / 111.0
                
                minx, miny, maxx, maxy = geometry.bounds
                buffered_bounds = (
                    minx - buffer_deg,
                    miny - buffer_deg,
                    maxx + buffer_deg,
                    maxy + buffer_deg
                )
                
                aoi_info['buffered_bounds'] = buffered_bounds
                aoi_info['buffer_km'] = buffer_km
                
                # Calculate center point
                centroid = geometry.centroid
                aoi_info['center_lat'] = centroid.y
                aoi_info['center_lon'] = centroid.x
                
                # Estimate DEM coverage area
                bbox_area_deg2 = (maxx - minx) * (maxy - miny)
                aoi_info['bbox_area_deg2'] = bbox_area_deg2
                
                processed_aois.append(aoi_info)
                
                logger.debug(f"Processed AOI: {aoi_info['study_site']} "
                           f"({aoi_info['area_km2']:.1f} km²)")
                
            except Exception as e:
                logger.error(f"Error processing AOI {idx} in {aoi_file_path}: {e}")
                continue
        
        return processed_aois
    
    def get_required_dem_tiles(self, aoi_list: List[Dict[str, Any]],
                             dem_source: str = "nasadem") -> List[str]:
        """
        Determine required DEM tiles for all AOIs.
        
        Parameters:
        -----------
        aoi_list : List[Dict[str, Any]]
            List of processed AOI information
        dem_source : str
            DEM source type
            
        Returns:
        --------
        List[str]
            List of required DEM tile names
        """
        from dem_downloader import DEMDownloader
        
        downloader = DEMDownloader(source=dem_source)
        all_tiles = set()
        
        logger.info(f"Determining DEM tiles for {len(aoi_list)} AOIs...")
        
        for aoi_info in aoi_list:
            try:
                # Get buffered bounds
                min_lon, min_lat, max_lon, max_lat = aoi_info['buffered_bounds']
                
                # Get required tiles for this AOI
                tiles = downloader.get_required_tiles_for_bbox(
                    min_lat, min_lon, max_lat, max_lon
                )
                
                all_tiles.update(tiles)
                
                logger.debug(f"AOI {aoi_info['study_site']}: {len(tiles)} tiles required")
                
            except Exception as e:
                logger.error(f"Error determining tiles for AOI {aoi_info.get('study_site', 'unknown')}: {e}")
                continue
        
        required_tiles = sorted(list(all_tiles))
        logger.info(f"Total unique DEM tiles required: {len(required_tiles)}")
        
        return required_tiles
    
    def create_combined_bbox(self, aoi_list: List[Dict[str, Any]],
                           buffer_km: float = 2.0) -> Tuple[float, float, float, float]:
        """
        Create combined bounding box for all AOIs.
        
        Parameters:
        -----------
        aoi_list : List[Dict[str, Any]]
            List of processed AOI information
        buffer_km : float
            Additional buffer in kilometers
            
        Returns:
        --------
        Tuple[float, float, float, float]
            Combined bounding box (min_lon, min_lat, max_lon, max_lat)
        """
        if not aoi_list:
            return (0, 0, 0, 0)
        
        # Get all bounds
        all_bounds = [aoi['buffered_bounds'] for aoi in aoi_list]
        
        # Find overall bounds
        min_lons = [bounds[0] for bounds in all_bounds]
        min_lats = [bounds[1] for bounds in all_bounds]
        max_lons = [bounds[2] for bounds in all_bounds]
        max_lats = [bounds[3] for bounds in all_bounds]
        
        overall_bounds = (
            min(min_lons),
            min(min_lats),
            max(max_lons),
            max(max_lats)
        )
        
        # Add additional buffer if requested
        if buffer_km > 0:
            buffer_deg = buffer_km / 111.0
            overall_bounds = (
                overall_bounds[0] - buffer_deg,
                overall_bounds[1] - buffer_deg,
                overall_bounds[2] + buffer_deg,
                overall_bounds[3] + buffer_deg
            )
        
        logger.info(f"Combined bbox: {overall_bounds[1]:.3f}°-{overall_bounds[3]:.3f}°N, "
                   f"{overall_bounds[0]:.3f}°-{overall_bounds[2]:.3f}°E")
        
        return overall_bounds
    
    def group_aois_by_region(self, aoi_list: List[Dict[str, Any]],
                           grid_size_deg: float = 1.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group AOIs by geographic region to optimize downloading.
        
        Parameters:
        -----------
        aoi_list : List[Dict[str, Any]]
            List of processed AOI information
        grid_size_deg : float
            Grid size in degrees for grouping
            
        Returns:
        --------
        Dict[str, List[Dict[str, Any]]]
            Dictionary of region_id -> list of AOIs
        """
        regions = {}
        
        for aoi_info in aoi_list:
            center_lat = aoi_info['center_lat']
            center_lon = aoi_info['center_lon']
            
            # Calculate grid cell
            grid_lat = math.floor(center_lat / grid_size_deg) * grid_size_deg
            grid_lon = math.floor(center_lon / grid_size_deg) * grid_size_deg
            
            region_id = f"{grid_lat:.1f}N_{grid_lon:.1f}E"
            
            if region_id not in regions:
                regions[region_id] = []
            
            regions[region_id].append(aoi_info)
        
        logger.info(f"Grouped {len(aoi_list)} AOIs into {len(regions)} regions")
        
        return regions
    
    def validate_aoi_coverage(self, aoi_list: List[Dict[str, Any]],
                            downloaded_tiles: List[str],
                            dem_source: str = "nasadem") -> Dict[str, Any]:
        """
        Validate that downloaded DEM tiles cover all AOIs.
        
        Parameters:
        -----------
        aoi_list : List[Dict[str, Any]]
            List of processed AOI information
        downloaded_tiles : List[str]
            List of successfully downloaded tile names
        dem_source : str
            DEM source type
            
        Returns:
        --------
        Dict[str, Any]
            Coverage validation results
        """
        from dem_downloader import DEMDownloader
        
        downloader = DEMDownloader(source=dem_source)
        
        coverage_results = {
            'total_aois': len(aoi_list),
            'fully_covered_aois': 0,
            'partially_covered_aois': 0,
            'uncovered_aois': 0,
            'coverage_details': []
        }
        
        for aoi_info in aoi_list:
            try:
                # Get required tiles for this AOI
                min_lon, min_lat, max_lon, max_lat = aoi_info['buffered_bounds']
                required_tiles = downloader.get_required_tiles_for_bbox(
                    min_lat, min_lon, max_lat, max_lon
                )
                
                # Check coverage
                available_tiles = [tile for tile in required_tiles if tile in downloaded_tiles]
                coverage_ratio = len(available_tiles) / len(required_tiles) if required_tiles else 0
                
                aoi_coverage = {
                    'study_site': aoi_info['study_site'],
                    'required_tiles': len(required_tiles),
                    'available_tiles': len(available_tiles),
                    'coverage_ratio': coverage_ratio,
                    'missing_tiles': [tile for tile in required_tiles if tile not in downloaded_tiles]
                }
                
                coverage_results['coverage_details'].append(aoi_coverage)
                
                # Categorize coverage
                if coverage_ratio >= 1.0:
                    coverage_results['fully_covered_aois'] += 1
                elif coverage_ratio > 0:
                    coverage_results['partially_covered_aois'] += 1
                else:
                    coverage_results['uncovered_aois'] += 1
                
            except Exception as e:
                logger.error(f"Error validating coverage for AOI {aoi_info.get('study_site', 'unknown')}: {e}")
                continue
        
        # Calculate overall coverage
        coverage_results['overall_coverage_ratio'] = (
            coverage_results['fully_covered_aois'] / coverage_results['total_aois']
            if coverage_results['total_aois'] > 0 else 0
        )
        
        logger.info(f"Coverage validation: {coverage_results['fully_covered_aois']}/{coverage_results['total_aois']} "
                   f"AOIs fully covered ({coverage_results['overall_coverage_ratio']*100:.1f}%)")
        
        return coverage_results
    
    def get_aoi_summary_stats(self, aoi_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics for AOI list."""
        if not aoi_list:
            return {}
        
        areas = [aoi['area_km2'] for aoi in aoi_list if aoi['area_km2'] > 0]
        center_lats = [aoi['center_lat'] for aoi in aoi_list]
        center_lons = [aoi['center_lon'] for aoi in aoi_list]
        
        stats = {
            'total_aois': len(aoi_list),
            'study_sites': list(set(aoi['study_site'] for aoi in aoi_list)),
            'total_area_km2': sum(areas),
            'avg_area_km2': sum(areas) / len(areas) if areas else 0,
            'min_area_km2': min(areas) if areas else 0,
            'max_area_km2': max(areas) if areas else 0,
            'spatial_extent': {
                'lat_range': (min(center_lats), max(center_lats)),
                'lon_range': (min(center_lons), max(center_lons)),
                'center': (
                    sum(center_lats) / len(center_lats),
                    sum(center_lons) / len(center_lons)
                )
            }
        }
        
        return stats
    
    def create_aoi_index_file(self, aoi_list: List[Dict[str, Any]],
                            output_file: Path) -> None:
        """Create an index file of all processed AOIs."""
        
        # Create simplified records for JSON serialization
        aoi_records = []
        for aoi in aoi_list:
            record = {
                'source_file': aoi['source_file'],
                'study_site': aoi['study_site'],
                'area_km2': aoi['area_km2'],
                'center_lat': aoi['center_lat'],
                'center_lon': aoi['center_lon'],
                'bounds': aoi['original_bounds'],
                'buffered_bounds': aoi['buffered_bounds'],
                'buffer_km': aoi['buffer_km']
            }
            aoi_records.append(record)
        
        # Create summary
        summary = self.get_aoi_summary_stats(aoi_list)
        
        index_data = {
            'summary': summary,
            'aois': aoi_records
        }
        
        # Save as JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(index_data, f, indent=2, default=str)
        
        logger.info(f"Created AOI index file: {output_file}")

# Utility functions
def load_and_process_aoi_file(aoi_file_path: str, buffer_km: float = 2.0) -> List[Dict[str, Any]]:
    """Load and process a single AOI file."""
    processor = AOIProcessor()
    aoi_gdf = gpd.read_file(aoi_file_path)
    return processor.process_aoi_file(aoi_file_path, aoi_gdf, buffer_km)

def discover_step2_aoi_files(step2_dir: str) -> List[str]:
    """Discover all AOI files in Step 2 directory."""
    step2_path = Path(step2_dir)
    
    aoi_files = []
    patterns = ["*aoi*.geojson", "*aoi*.shp", "*AOI*.geojson", "*AOI*.shp"]
    
    for pattern in patterns:
        aoi_files.extend(step2_path.rglob(pattern))
    
    return [str(f) for f in aoi_files]

# Example usage
if __name__ == "__main__":
    # Test AOI processor
    print("📐 Testing AOI Processor")
    print("=" * 50)
    
    processor = AOIProcessor()
    
    # Test with synthetic AOI
    from shapely.geometry import Polygon
    
    # Create test AOI (Cameroon region)
    test_polygon = Polygon([
        (12.5, 7.5),   # SW
        (13.5, 7.5),   # SE
        (13.5, 8.5),   # NE
        (12.5, 8.5),   # NW
        (12.5, 7.5)    # Close
    ])
    
    test_gdf = gpd.GeoDataFrame(
        [{
            'study_site': 'Test Site',
            'area_km2': 100.0
        }],
        geometry=[test_polygon],
        crs='EPSG:4326'
    )
    
    # Process test AOI
    processed = processor.process_aoi_file("test_aoi.geojson", test_gdf, buffer_km=2.0)
    print(f"Processed {len(processed)} test AOIs")
    
    # Test tile determination
    required_tiles = processor.get_required_dem_tiles(processed, "nasadem")
    print(f"Required tiles for test AOI: {required_tiles}")
    
    # Test combined bbox
    combined_bbox = processor.create_combined_bbox(processed)
    print(f"Combined bounding box: {combined_bbox}")
    
    # Test summary stats
    stats = processor.get_aoi_summary_stats(processed)
    print(f"Summary stats: {stats}")
    
    print("\n🎉 AOI processor working correctly!")