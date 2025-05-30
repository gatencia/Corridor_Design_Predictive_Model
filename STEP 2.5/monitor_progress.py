#!/usr/bin/env python3
"""
Real-time progress monitor for STEP 2.5
Run this in a separate terminal while download is running
"""

import time
import os
from pathlib import Path
from datetime import datetime, timedelta

def monitor_progress():
    """Monitor download progress in real-time."""
    
    print("📊 STEP 2.5 Progress Monitor")
    print("=" * 50)
    
    output_dir = Path("../STEP 3/data/raw/dem")
    tiles_dir = output_dir / "tiles"
    mosaics_dir = output_dir / "mosaics"
    logs_dir = Path("logs")
    
    start_time = time.time()
    last_tile_count = 0
    last_mosaic_count = 0
    
    print(f"Monitoring: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            # Count downloaded tiles
            tile_count = 0
            if tiles_dir.exists():
                tile_count = len(list(tiles_dir.rglob("*.zip")))
            
            # Count created mosaics
            mosaic_count = 0
            if mosaics_dir.exists():
                mosaic_count = len(list(mosaics_dir.glob("*.tif")))
            
            # Calculate download rate
            tiles_downloaded = tile_count - last_tile_count
            if tiles_downloaded > 0:
                download_rate = tiles_downloaded / 30  # tiles per second (30-sec interval)
            else:
                download_rate = 0
            
            # Show progress
            print(f"\r🕐 {datetime.now().strftime('%H:%M:%S')} | "
                  f"⏱️  {elapsed_str} | "
                  f"📦 Tiles: {tile_count} | "
                  f"🗺️  Mosaics: {mosaic_count} | "
                  f"⚡ Rate: {download_rate:.1f}/30s", end="")
            
            # Show milestone updates
            if tile_count > last_tile_count:
                print(f"\n   ✅ Downloaded {tile_count} tiles (+{tiles_downloaded} in last 30s)")
            
            if mosaic_count > last_mosaic_count:
                new_mosaics = mosaic_count - last_mosaic_count
                print(f"\n   🗺️  Created {mosaic_count} mosaics (+{new_mosaics})")
            
            last_tile_count = tile_count
            last_mosaic_count = mosaic_count
            
            # Check if process is likely complete
            if mosaic_count >= 24:  # All collar AOIs should have mosaics
                print(f"\n\n🎉 Process appears complete!")
                print(f"   Final counts: {tile_count} tiles, {mosaic_count} mosaics")
                print(f"   Total time: {elapsed_str}")
                break
            
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Monitoring stopped")
        print(f"   Final counts: {tile_count} tiles, {mosaic_count} mosaics")
        print(f"   Elapsed time: {elapsed_str}")

def show_current_status():
    """Show current status without continuous monitoring."""
    
    output_dir = Path("../STEP 3/data/raw/dem")
    
    print("📊 Current Status")
    print("=" * 30)
    
    # Check tiles
    tiles_dir = output_dir / "tiles"
    if tiles_dir.exists():
        tile_count = len(list(tiles_dir.rglob("*.zip")))
        tile_size = sum(f.stat().st_size for f in tiles_dir.rglob("*") if f.is_file()) / (1024*1024)
        print(f"📦 Downloaded tiles: {tile_count}")
        print(f"💾 Tiles size: {tile_size:.1f} MB")
    else:
        print("📦 No tiles directory yet")
    
    # Check mosaics
    mosaics_dir = output_dir / "mosaics" 
    if mosaics_dir.exists():
        mosaic_files = list(mosaics_dir.glob("*.tif"))
        mosaic_count = len(mosaic_files)
        
        if mosaic_files:
            mosaic_size = sum(f.stat().st_size for f in mosaic_files) / (1024*1024)
            print(f"🗺️  Created mosaics: {mosaic_count}")
            print(f"💾 Mosaics size: {mosaic_size:.1f} MB")
            
            print(f"\nMosaics created:")
            for mosaic in sorted(mosaic_files):
                size_mb = mosaic.stat().st_size / (1024*1024)
                print(f"   • {mosaic.name} ({size_mb:.1f} MB)")
        else:
            print("🗺️  Mosaics directory exists but empty")
    else:
        print("🗺️  No mosaics directory yet")
    
    # Check logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"\n📄 Latest log: {latest_log.name}")
            
            # Show last few lines
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("   Last log entries:")
                        for line in lines[-3:]:
                            print(f"   {line.strip()}")
            except:
                pass
    
    print(f"\n🎯 Target: 24 collar mosaics")
    
    if mosaic_count >= 24:
        print("✅ Process appears complete!")
    elif tile_count > 0:
        print("🔄 Download in progress...")
    else:
        print("⏳ Process not started or just beginning...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        show_current_status()
    else:
        monitor_progress()