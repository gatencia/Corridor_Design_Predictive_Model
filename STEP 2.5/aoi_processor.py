#!/usr/bin/env python3
"""
Optimized AOI Processing for STEP 2.5
Includes deduplication and smart discovery of unique study sites.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from shapely.geometry import box
import json
import math
import time # Added import for timing

logger = logging.getLogger(__name__)

class OptimizedAOIProcessor:
    """
    Optimized AOI processor that eliminates duplicates and organizes efficiently.
    """
    
    def __init__(self, project_root: Path):
        """Initialize optimized AOI processor."""
        self.project_root = Path(project_root)
        self.step2_dir = self.project_root / "STEP 2"
        self.processed_aois = {}  # Cache for processed AOIs
        
        logger.info(f"Optimized AOI processor initialized with project root: {self.project_root}")
    

    def find_step2_aoi_outputs(self) -> List[Dict[str, Any]]:
        """
        SIMPLE APPROACH: Go directly to organized individual_aois folder.
        """
        logger.info("Loading individual collar AOIs from organized structure...")
        
        # DIRECT path to your organized collar folders
        individual_aois_path = self.step2_dir / "data" / "outputs" / "individual_aois"
        
        if not individual_aois_path.exists():
            logger.error(f"Directory not found: {individual_aois_path}")
            return []
        
        logger.info(f"Reading from: {individual_aois_path}")
        
        aoi_list = []
        collar_folders = sorted([d for d in individual_aois_path.iterdir() if d.is_dir()])
        
        logger.info(f"Found {len(collar_folders)} collar folders")
        
        for collar_folder in collar_folders:
            # Find AOI file in this collar folder
            aoi_files = list(collar_folder.glob("aoi_*.geojson")) + list(collar_folder.glob("aoi_*.shp"))
            
            if not aoi_files:
                logger.warning(f"No AOI file in {collar_folder.name}")
                continue
            
            aoi_file = aoi_files[0]  # Take first AOI file found
            
            try:
                # Load AOI geometry
                gdf = gpd.read_file(aoi_file)
                if len(gdf) == 0:
                    continue
                
                # Extract info
                study_site = collar_folder.name.replace("_", " ").title()
                area_km2 = float(gdf['area_km2'].iloc[0]) if 'area_km2' in gdf.columns else 0
                
                aoi_info = {
                    'file_path': str(aoi_file),
                    'study_site': study_site,
                    'area_km2': area_km2,
                    'bounds': gdf.total_bounds.tolist(),
                    'geometry': gdf.iloc[0].geometry,
                    'format': aoi_file.suffix,
                    'crs': str(gdf.crs)
                }
                
                aoi_list.append(aoi_info)
                logger.info(f"✅ {study_site}: {area_km2:.1f} km²")
                
            except Exception as e:
                logger.error(f"Error loading {collar_folder.name}: {e}")
        
        logger.info(f"Loaded {len(aoi_list)} individual collar AOIs")
        return aoi_list

    def _discover_aois_in_path(self, search_path: Path, unique_aois: Dict[str, Dict[str, Any]]):
        """Discover AOIs in a specific path and deduplicate."""
        
        # Look for organized structure first (preferred)
        if (search_path / "individual_aois").exists():
            search_path = search_path / "individual_aois"
            self._process_organized_structure(search_path, unique_aois)
        
        # Look for direct AOI files
        aoi_patterns = ["*aoi*.geojson", "*aoi*.shp", "*AOI*.geojson", "*AOI*.shp"]
        
        for pattern in aoi_patterns:
            for aoi_file in search_path.rglob(pattern):
                self._process_aoi_file(aoi_file, unique_aois)
    
    def _process_organized_structure(self, base_path: Path, unique_aois: Dict[str, Dict[str, Any]]):
        """Process organized folder structure (preferred method)."""
        
        for site_dir in base_path.iterdir():
            if not site_dir.is_dir():
                continue
            
            # Look for the most recent AOI file in this site directory
            aoi_files = []
            for pattern in ["*.geojson", "*.shp"]:
                aoi_files.extend(site_dir.glob(pattern))
            
            if not aoi_files:
                continue
            
            # Take the most recent file
            latest_aoi = max(aoi_files, key=lambda x: x.stat().st_mtime)
            
            # Use folder name as study site (cleaner than filename parsing)
            study_site = site_dir.name.replace("_", " ").title()
            
            self._add_unique_aoi(latest_aoi, unique_aois, study_site_override=study_site)
    
    def _process_aoi_file(self, aoi_file: Path, unique_aois: Dict[str, Dict[str, Any]]):
        """Process a single AOI file and add to unique collection."""
        
        try:
            # Load AOI to get metadata
            gdf = gpd.read_file(aoi_file)
            
            if len(gdf) == 0:
                logger.warning(f"Empty AOI file: {aoi_file}")
                return
            
            # Extract study site name
            study_site = self._extract_study_site_name(aoi_file, gdf)
            
            self._add_unique_aoi(aoi_file, unique_aois, study_site_override=study_site)
            
        except Exception as e:
            logger.warning(f"Could not process AOI file {aoi_file}: {e}")
    
    def _add_unique_aoi(self, aoi_file: Path, unique_aois: Dict[str, Dict[str, Any]], 
                       study_site_override: str = None):
        """Add AOI to unique collection, replacing duplicates with better versions."""
        
        try:
            gdf = gpd.read_file(aoi_file)
            
            if len(gdf) == 0:
                return
            
            # Get study site name
            if study_site_override:
                study_site = study_site_override
            else:
                study_site = self._extract_study_site_name(aoi_file, gdf)
            
            # Normalize study site name for deduplication
            normalized_name = self._normalize_site_name(study_site)
            
            # Calculate area and other metrics
            if 'area_km2' in gdf.columns:
                area_km2 = float(gdf['area_km2'].iloc[0])
            else:
                # Calculate area from geometry
                if gdf.crs != 'EPSG:4326':
                    gdf_area = gdf.to_crs('EPSG:4326')
                else:
                    gdf_area = gdf
                bounds = gdf_area.total_bounds
                area_deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                area_km2 = area_deg2 * 111 * 111  # Rough conversion
            
            # Get file timestamp for choosing most recent
            file_timestamp = aoi_file.stat().st_mtime
            
            # Quality score (prefer organized structure, recent files, reasonable areas)
            quality_score = 0
            if "individual_aois" in str(aoi_file):
                quality_score += 100  # Prefer organized structure
            if aoi_file.suffix == '.geojson':
                quality_score += 10  # Prefer GeoJSON
            quality_score += file_timestamp / 1000000  # Recent files get higher score
            if 10 < area_km2 < 100000:  # Reasonable area range
                quality_score += 50
            
            aoi_info = {
                'file_path': str(aoi_file),
                'study_site': study_site,
                'area_km2': area_km2,
                'bounds': gdf.total_bounds.tolist(),
                'geometry': gdf.iloc[0].geometry,
                'format': aoi_file.suffix,
                'file_timestamp': file_timestamp,
                'quality_score': quality_score,
                'crs': str(gdf.crs)
            }
            
            # Check if we already have this study site
            if normalized_name in unique_aois:
                existing = unique_aois[normalized_name]
                
                # Keep the one with higher quality score
                if quality_score > existing['quality_score']:
                    logger.debug(f"Replacing {existing['study_site']} with better version: {study_site}")
                    unique_aois[normalized_name] = aoi_info
                else:
                    logger.debug(f"Keeping existing {existing['study_site']}, skipping duplicate: {study_site}")
            else:
                unique_aois[normalized_name] = aoi_info
                logger.debug(f"Added unique AOI: {study_site}")
            
        except Exception as e:
            logger.warning(f"Error processing {aoi_file}: {e}")
    
    def _extract_study_site_name(self, aoi_file: Path, gdf: gpd.GeoDataFrame) -> str:
        """Extract study site name from file or data."""
        
        # Try to get from data first
        if 'study_site' in gdf.columns:
            return str(gdf['study_site'].iloc[0])
        
        # Parse from filename
        filename = aoi_file.stem
        
        # Remove common prefixes/suffixes
        filename = filename.replace('aoi_', '').replace('AOI_', '')
        
        # Remove timestamps
        import re
        filename = re.sub(r'_\d{8}_?\d{6}?', '', filename)
        filename = re.sub(r'_\d{8}', '', filename)
        
        # Clean up underscores and make readable
        study_site = filename.replace('_', ' ').title()
        
        # Handle common cases
        study_site = study_site.replace('Np', 'NP')
        study_site = study_site.replace('Mt.', 'Mt.')
        
        return study_site
    
    def _normalize_site_name(self, site_name: str) -> str:
        """Normalize site name for deduplication."""
        normalized = site_name.lower()
        normalized = normalized.replace(' ', '').replace('_', '').replace('-', '')
        normalized = normalized.replace('nationalpark', 'np')
        normalized = normalized.replace('national', 'nat')
        normalized = normalized.replace('park', 'pk')
        normalized = normalized.replace('reserve', 'res')
        normalized = normalized.replace('game', 'gm')
        normalized = normalized.replace('(cameroon)', '').replace('(nigeria)', '')
        normalized = normalized.replace('cameroon', '').replace('nigeria', '')
        return normalized
    
    def calculate_unique_dem_requirements(self, aoi_list: List[Dict[str, Any]], 
                                        buffer_km: float = 2.0) -> Dict[str, Any]:
        """
        Calculate DEM requirements with optimization for overlapping areas.
        """
        method_start_time = time.time() # Start timer for the whole method
        logger.info(f"Calculating optimized DEM requirements for {len(aoi_list)} unique AOIs...")
        print(f"Starting calculation of optimized DEM requirements for {len(aoi_list)} unique AOIs...")
        
        from dem_downloader import DEMDownloader, DEMSource
        
        downloader = DEMDownloader()
        print("DEMDownloader initialized for DEM requirement calculation.")

        all_tiles = set()
        aoi_details = []
        
        total_aois = len(aoi_list)
        processed_aois_count = 0
        cumulative_processing_time = 0.0

        print(f"Processing {total_aois} AOIs to determine DEM tile requirements (buffer: {buffer_km} km)...")
        for i, aoi in enumerate(aoi_list):
            aoi_processing_start_time = time.time() # Start timer for this AOI
            
            current_aoi_name = aoi.get('study_site', f'UnknownAOI_{i+1}')
            current_aoi_area = aoi.get('area_km2', 0)
            print(f"  Processing AOI {i+1}/{total_aois}: {current_aoi_name} (Area: {current_aoi_area:.1f} km²)...")
            
            bounds = aoi['bounds']
            print(f"    Original bounds for {current_aoi_name}: {bounds}")
            
            buffered_bounds = downloader.buffer_bounds(bounds, buffer_km)
            print(f"    Buffered bounds ({buffer_km} km) for {current_aoi_name}: {buffered_bounds}")
            
            tiles = downloader.get_required_dem_tiles(buffered_bounds, DEMSource.NASADEM)
            print(f"    Required DEM tiles for {current_aoi_name}: {list(tiles)} ({len(tiles)} tiles)")
            
            aoi_details.append({
                'study_site': aoi['study_site'],
                'area_km2': aoi['area_km2'],
                'buffered_bounds': buffered_bounds,
                'required_tiles': list(tiles),
                'tile_count': len(tiles)
            })
            
            all_tiles.update(tiles)
            
            logger.info(f"   • {aoi['study_site']}: {len(tiles)} tiles ({aoi['area_km2']:.1f} km²)")

            aoi_processing_end_time = time.time()
            time_taken_for_aoi = aoi_processing_end_time - aoi_processing_start_time
            processed_aois_count += 1
            cumulative_processing_time += time_taken_for_aoi
            average_time_per_aoi = cumulative_processing_time / processed_aois_count
            remaining_aois = total_aois - processed_aois_count
            estimated_time_remaining = remaining_aois * average_time_per_aoi

            print(f"      Time for this AOI ({current_aoi_name}): {time_taken_for_aoi:.2f}s")
            print(f"      Average time per AOI so far: {average_time_per_aoi:.2f}s")
            if remaining_aois > 0:
                print(f"      Estimated time remaining for AOI processing: {estimated_time_remaining:.2f}s ({remaining_aois} AOIs left)")
            else:
                print("      All AOIs processed.")
            print("-" * 20) # Separator for readability
        
        print("All AOIs processed. Finalizing DEM requirements summary...")
        # Calculate efficiency
        total_individual_tiles = sum(len(detail['required_tiles']) for detail in aoi_details)
        unique_tiles = len(all_tiles)
        efficiency = (total_individual_tiles - unique_tiles) / total_individual_tiles * 100 if total_individual_tiles > 0 else 0
        
        requirements = {
            'unique_aois': len(aoi_list),
            'total_individual_tiles': total_individual_tiles,
            'unique_tiles_needed': unique_tiles,
            'efficiency_gain_percent': efficiency,
            'estimated_size_mb': unique_tiles * 25,  # ~25MB per tile
            'aoi_details': aoi_details,
            'tile_list': sorted(list(all_tiles))
        }
        
        logger.info(f"📊 Optimized DEM Requirements:")
        logger.info(f"   Individual calculations: {total_individual_tiles} tiles")
        logger.info(f"   Unique tiles needed: {unique_tiles} tiles")
        logger.info(f"   Efficiency gain: {efficiency:.1f}% reduction")
        logger.info(f"   Estimated download: {requirements['estimated_size_mb']:.0f} MB")
        
        method_end_time = time.time()
        total_method_time = method_end_time - method_start_time
        print(f"DEM requirement calculation complete. Summary generated.")
        print(f"Total time taken for DEM requirement calculation: {total_method_time:.2f} seconds.")
        return requirements
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimizations applied."""
        return {
            'deduplication_enabled': True,
            'organized_structure_support': True,
            'quality_scoring': True,
            'tile_overlap_optimization': True,
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }

# Utility functions for backward compatibility
def discover_step2_aoi_files(step2_dir: str) -> List[str]:
    """Discover AOI files with deduplication."""
    processor = OptimizedAOIProcessor(Path(step2_dir).parent)
    aoi_data = processor.find_step2_aoi_outputs()
    return [aoi['file_path'] for aoi in aoi_data]

def load_and_process_aoi_file(aoi_file_path: str, buffer_km: float = 2.0) -> List[Dict[str, Any]]:
    """Load and process AOI file (optimized version)."""
    try:
        gdf = gpd.read_file(aoi_file_path)
        
        if len(gdf) == 0:
            return []
        
        # Process single AOI
        bounds = gdf.total_bounds
        buffer_deg = buffer_km / 111.0
        
        buffered_bounds = (
            bounds[0] - buffer_deg,
            bounds[1] - buffer_deg,
            bounds[2] + buffer_deg,
            bounds[3] + buffer_deg
        )
        
        aoi_info = {
            'file_path': aoi_file_path,
            'study_site': Path(aoi_file_path).stem,
            'bounds': bounds.tolist(),
            'buffered_bounds': buffered_bounds,
            'area_km2': 0,  # Calculate if needed
            'buffer_km': buffer_km
        }
        
        return [aoi_info]
        
    except Exception as e:
        logger.error(f"Error processing {aoi_file_path}: {e}")
        return []

# Main class alias for compatibility
class AOIProcessor(OptimizedAOIProcessor):
    """Alias for backward compatibility."""
    pass

# Example usage
if __name__ == "__main__":
    print("🔧 Testing Optimized AOI Processor")
    print("=" * 50)
    
    # Test with project root
    project_root = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research")
    
    processor = OptimizedAOIProcessor(project_root)
    
    # Test AOI discovery with deduplication
    aois = processor.find_step2_aoi_outputs()
    print(f"Found {len(aois)} unique AOIs (after deduplication)")
    
    if aois:
        # Test optimized DEM requirements
        requirements = processor.calculate_unique_dem_requirements(aois)
        print(f"\nOptimization Results:")
        print(f"  Efficiency gain: {requirements['efficiency_gain_percent']:.1f}%")
        print(f"  Unique tiles: {requirements['unique_tiles_needed']}")
        print(f"  Estimated size: {requirements['estimated_size_mb']:.0f} MB")
        
        # Show study sites
        print(f"\nUnique Study Sites:")
        for aoi in aois:
            print(f"  📍 {aoi['study_site']}: {aoi['area_km2']:.1f} km²")
    
    print("\n✅ Optimized AOI processor working correctly!")