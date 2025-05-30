#!/usr/bin/env python3
"""
Quick fix for aoi_processor.py - save as fix_aoi_processor.py in STEP 2.5/
"""

from pathlib import Path
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)

class QuickAOIProcessor:
    """Simple, direct AOI processor for timing tests."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.step2_dir = self.project_root / "STEP 2"
    
    def find_step2_aoi_outputs(self):
        """Go directly to organized individual_aois folder."""
        print("🔍 Loading collar AOIs from organized structure...")
        
        # Direct path to your 24 collar folders
        individual_aois_path = self.step2_dir / "data" / "outputs" / "individual_aois"
        
        if not individual_aois_path.exists():
            print(f"❌ Directory not found: {individual_aois_path}")
            return []
        
        print(f"📁 Reading from: {individual_aois_path}")
        
        aoi_list = []
        collar_folders = sorted([d for d in individual_aois_path.iterdir() if d.is_dir()])
        
        print(f"📊 Found {len(collar_folders)} collar folders")
        
        for i, collar_folder in enumerate(collar_folders, 1):
            print(f"   {i:2d}/24 Processing: {collar_folder.name}")
            
            # Find AOI file in this collar folder
            aoi_files = list(collar_folder.glob("aoi_*.geojson")) + list(collar_folder.glob("aoi_*.shp"))
            
            if not aoi_files:
                print(f"      ⚠️  No AOI file found")
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
                bounds = gdf.total_bounds.tolist()  # [min_lon, min_lat, max_lon, max_lat]
                
                aoi_info = {
                    'file_path': str(aoi_file),
                    'study_site': study_site,
                    'area_km2': area_km2,
                    'bounds': bounds,
                    'geometry': gdf.iloc[0].geometry,
                    'format': aoi_file.suffix,
                    'crs': str(gdf.crs)
                }
                
                aoi_list.append(aoi_info)
                print(f"      ✅ {area_km2:.1f} km² ({aoi_file.suffix})")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        print(f"\n✅ Successfully loaded {len(aoi_list)} collar AOIs")
        return aoi_list

# Test function
def test_aoi_loading():
    """Test AOI loading and show what we'll download."""
    print("🧪 Testing AOI Loading")
    print("=" * 50)
    
    project_root = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research")
    processor = QuickAOIProcessor(project_root)
    
    aois = processor.find_step2_aoi_outputs()
    
    if not aois:
        print("❌ No AOIs found!")
        return
    
    print(f"\n📊 Summary:")
    print(f"   Collar AOIs found: {len(aois)}")
    total_area = sum(aoi['area_km2'] for aoi in aois)
    print(f"   Total study area: {total_area:,.1f} km²")
    
    # Show first few
    print(f"\n📍 First 5 AOIs:")
    for i, aoi in enumerate(aois[:5], 1):
        print(f"   {i}. {aoi['study_site']}: {aoi['area_km2']:.1f} km²")
    
    return aois

if __name__ == "__main__":
    test_aoi_loading()