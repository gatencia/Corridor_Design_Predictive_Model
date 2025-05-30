"""
Main execution script for Step 2.5 DEM acquisition.
Simple wrapper to run the full pipeline.
"""

def main():
    """Main execution function."""
    print("🏔️  Executing Step 2.5: DEM Acquisition")
    print("Downloading SRTM tiles for all Step 2 AOIs...")
    print()
    
    try:
        # Import and run the main DEM downloader
        from step25_dem_downloader import main as dem_main
        
        success = dem_main()
        
        if success:
            print("\n✅ Step 2.5 completed successfully!")
            print("DEM tiles are ready for Step 3 (EnergyScape) processing.")
        else:
            print("\n❌ Step 2.5 completed with errors.")
            print("Check logs for details on failed downloads.")
        
        return success
        
    except ImportError as e:
        print(f"❌ Could not import DEM downloader: {e}")
        print("Ensure step25_dem_downloader.py is in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
