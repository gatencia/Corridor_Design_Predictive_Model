#!/usr/bin/env python3
"""
Quick cleanup script for Step 2 outputs
Removes duplicate and old processing files.
"""

import shutil
from pathlib import Path
import argparse
from datetime import datetime

def cleanup_step2_outputs(dry_run=False):
    """Clean up Step 2 outputs to remove duplicates."""
    
    print("🧹 Step 2 Output Cleanup Tool")
    print("=" * 50)
    
    step2_dir = Path("../STEP 2") if Path("../STEP 2").exists() else Path("STEP 2")
    
    if not step2_dir.exists():
        print("❌ Step 2 directory not found")
        return
    
    # Directories to clean
    cleanup_targets = [
        step2_dir / "data" / "outputs",
        step2_dir / "data" / "processed", 
        step2_dir / "outputs",
        step2_dir / "reports"
    ]
    
    total_size = 0
    total_files = 0
    
    print(f"🔍 Scanning for files to clean...")
    
    for target_dir in cleanup_targets:
        if target_dir.exists():
            files = list(target_dir.rglob("*"))
            files = [f for f in files if f.is_file()]
            
            dir_size = sum(f.stat().st_size for f in files)
            total_size += dir_size
            total_files += len(files)
            
            print(f"   📁 {target_dir.name}: {len(files)} files ({dir_size / (1024*1024):.1f} MB)")
    
    print(f"\n📊 Total to clean: {total_files} files ({total_size / (1024*1024):.1f} MB)")
    
    if dry_run:
        print("\n🔍 DRY RUN - No files will be deleted")
        print("Run without --dry-run to actually clean files")
        return
    
    # Confirm deletion
    response = input(f"\n⚠️  Delete all {total_files} files? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled")
        return
    
    # Perform cleanup
    print(f"\n🗑️  Cleaning up files...")
    
    cleaned_dirs = 0
    for target_dir in cleanup_targets:
        if target_dir.exists():
            try:
                shutil.rmtree(target_dir)
                print(f"   ✅ Cleaned: {target_dir}")
                cleaned_dirs += 1
            except Exception as e:
                print(f"   ❌ Error cleaning {target_dir}: {e}")
    
    print(f"\n✅ Cleanup complete! Cleaned {cleaned_dirs} directories")
    print(f"💾 Freed up ~{total_size / (1024*1024):.1f} MB of space")
    print(f"\n💡 Now run the optimized Step 2 processing:")
    print(f"   cd '{step2_dir}' && python run_gps_processing.py")

def main():
    parser = argparse.ArgumentParser(description="Clean up Step 2 outputs")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be cleaned without deleting')
    
    args = parser.parse_args()
    cleanup_step2_outputs(args.dry_run)

if __name__ == "__main__":
    main()