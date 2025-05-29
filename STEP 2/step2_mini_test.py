#!/usr/bin/env python3
"""
MINIMAL test to isolate the hanging issue.
"""

print("🔍 MINIMAL TEST: Starting...")

import sys
from pathlib import Path

print("🔍 MINIMAL TEST: Basic imports done")

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

print("🔍 MINIMAL TEST: Path setup done")

try:
    print("🔍 MINIMAL TEST: Attempting to import GPSDataProcessor...")
    from data_ingestion import GPSDataProcessor
    print("✅ MINIMAL TEST: GPSDataProcessor imported successfully")
except Exception as e:
    print(f"❌ MINIMAL TEST: Import failed: {e}")
    sys.exit(1)

try:
    print("🔍 MINIMAL TEST: Creating GPSDataProcessor instance...")
    processor = GPSDataProcessor()
    print("✅ MINIMAL TEST: GPSDataProcessor created successfully")
except Exception as e:
    print(f"❌ MINIMAL TEST: Creation failed: {e}")
    sys.exit(1)

try:
    print("🔍 MINIMAL TEST: Finding GPS data...")
    data_dir = Path("../GPS_Collar_CSV_Mark")
    if not data_dir.exists():
        data_dir = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark")
    
    if not data_dir.exists():
        print("❌ MINIMAL TEST: No data directory found")
        sys.exit(1)
    
    print(f"✅ MINIMAL TEST: Found data directory: {data_dir}")
    
    gps_files = list(data_dir.glob("*.csv"))
    print(f"✅ MINIMAL TEST: Found {len(gps_files)} CSV files")
    
    if not gps_files:
        print("❌ MINIMAL TEST: No CSV files found")
        sys.exit(1)
    
    # Test with just the smallest file
    file_sizes = [(f, f.stat().st_size) for f in gps_files]
    smallest_file = min(file_sizes, key=lambda x: x[1])[0]
    
    print(f"🔍 MINIMAL TEST: Testing with smallest file: {smallest_file.name} ({smallest_file.stat().st_size / 1024:.1f} KB)")
    
except Exception as e:
    print(f"❌ MINIMAL TEST: File discovery failed: {e}")
    sys.exit(1)

try:
    print("🔍 MINIMAL TEST: Attempting to load ONE small GPS file...")
    gdf = processor.load_gps_data(smallest_file)
    print(f"✅ MINIMAL TEST: Successfully loaded {len(gdf)} GPS points")
    
except Exception as e:
    print(f"❌ MINIMAL TEST: GPS loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("🎉 MINIMAL TEST: Everything works! The issue is elsewhere.")
print("Your GPS processing should work fine.")