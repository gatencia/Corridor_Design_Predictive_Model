#!/usr/bin/env python3
"""
Timing Test for STEP 2.5 Local Processing
Run this to get baseline performance metrics.
"""

import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import requests

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_internet_speed():
    """Quick internet speed test."""
    print("🌐 Testing internet speed...")
    
    test_url = "https://httpbin.org/bytes/1048576"  # 1MB test file
    
    try:
        start_time = time.time()
        response = requests.get(test_url, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            download_time = end_time - start_time
            file_size_mb = len(response.content) / (1024 * 1024)
            speed_mbps = (file_size_mb * 8) / download_time  # Convert to Mbps
            
            print(f"   📊 Download speed: ~{speed_mbps:.1f} Mbps")
            print(f"   ⏱️  1MB downloaded in {download_time:.2f} seconds")
            
            return speed_mbps
        else:
            print("   ⚠️  Speed test failed")
            return 0
    except Exception as e:
        print(f"   ❌ Speed test error: {e}")
        return 0

def estimate_download_time(num_tiles, avg_tile_size_mb=25, speed_mbps=50):
    """Estimate total download time."""
    
    if speed_mbps <= 0:
        speed_mbps = 50  # Assume 50 Mbps if test failed
    
    # Convert Mbps to MB/s
    speed_mb_per_sec = speed_mbps / 8
    
    # Calculate time per tile
    time_per_tile = avg_tile_size_mb / speed_mb_per_sec
    
    # Total time (including overhead)
    total_time_seconds = (time_per_tile * num_tiles) * 1.3  # 30% overhead
    
    return total_time_seconds

def run_timing_test():
    """Run complete timing test."""
    
    print("⏱️  STEP 2.5 Local Timing Test")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # 1. Test internet speed
    speed = test_internet_speed()
    
    # 2. Test AOI loading
    print(f"\n📁 Testing AOI loading...")
    try:
        from fix_aoi_processor import QuickAOIProcessor
        
        project_root = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research")
        processor = QuickAOIProcessor(project_root)
        
        aoi_start = time.time()
        aois = processor.find_step2_aoi_outputs()
        aoi_time = time.time() - aoi_start
        
        print(f"   ✅ AOI loading: {aoi_time:.2f} seconds")
        print(f"   📊 Found {len(aois)} collar AOIs")
        
        if len(aois) != 24:
            print(f"   ⚠️  Expected 24 AOIs, found {len(aois)}")
        
    except Exception as e:
        print(f"   ❌ AOI loading failed: {e}")
        return
    
    # 3. Estimate DEM requirements
    print(f"\n📐 Calculating DEM requirements...")
    
    try:
        from dem_downloader import DEMDownloader, DEMSource
        
        downloader = DEMDownloader()
        all_tiles = set()
        
        calc_start = time.time()
        
        for aoi in aois:
            bounds = aoi['bounds']
            buffered_bounds = downloader.buffer_bounds(bounds, 2.0)  # 2km buffer
            tiles = downloader.get_required_dem_tiles(buffered_bounds, DEMSource.NASADEM)
            all_tiles.update(tiles)
        
        calc_time = time.time() - calc_start
        
        print(f"   ✅ Calculation time: {calc_time:.2f} seconds")
        print(f"   📦 Unique tiles needed: {len(all_tiles)}")
        print(f"   💾 Estimated download size: {len(all_tiles) * 25:.0f} MB")
        
        # 4. Time estimates
        print(f"\n⏱️  Timing Estimates:")
        
        estimated_time = estimate_download_time(len(all_tiles), 25, speed)
        estimated_hours = estimated_time / 3600
        estimated_minutes = estimated_time / 60
        
        print(f"   📡 Your internet speed: ~{speed:.1f} Mbps")
        print(f"   📦 Download size: ~{len(all_tiles) * 25:.0f} MB")
        print(f"   ⏱️  Estimated download time: {estimated_minutes:.1f} minutes ({estimated_hours:.1f} hours)")
        
        # Show tile breakdown by region
        print(f"\n📍 Sample tiles needed:")
        sample_tiles = sorted(list(all_tiles))[:10]
        for i, tile in enumerate(sample_tiles, 1):
            print(f"   {i:2d}. {tile}")
        if len(all_tiles) > 10:
            print(f"   ... and {len(all_tiles) - 10} more tiles")
        
        # Final recommendations
        print(f"\n💡 Recommendations:")
        
        if estimated_hours < 1:
            print(f"   ✅ Local processing looks fast enough (<1 hour)")
            print(f"   💰 AWS probably not worth the setup cost")
        elif estimated_hours < 2:
            print(f"   ⚖️  Borderline case (1-2 hours)")
            print(f"   🤔 AWS might save ~1 hour for ~$10-15")
        else:
            print(f"   🚀 AWS would likely be much faster")
            print(f"   💰 Could reduce {estimated_hours:.1f}h to ~0.5h for ~$10-15")
        
        return {
            'aoi_count': len(aois),
            'tile_count': len(all_tiles),
            'download_size_mb': len(all_tiles) * 25,
            'internet_speed_mbps': speed,
            'estimated_time_minutes': estimated_minutes,
            'estimated_time_hours': estimated_hours
        }
        
    except Exception as e:
        print(f"   ❌ DEM calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_timing_test()