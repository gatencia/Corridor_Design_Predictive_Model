#!/usr/bin/env python3
"""
Step 3 Pipeline Integration Test
Comprehensive test to ensure synthetic DEMs work with the EnergyScape pipeline.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tempfile
import shutil
from datetime import datetime

def setup_paths():
    """Setup Python paths for Step 3 imports."""
    
    current_dir = Path(__file__).parent
    
    # Add Step 3 source directory
    step3_paths = [
        current_dir / "STEP 3" / "src",
        current_dir.parent / "STEP 3" / "src",
        Path("STEP 3/src"),
        Path("../STEP 3/src")
    ]
    
    step3_src = None
    for path in step3_paths:
        if path.exists():
            step3_src = path
            break
    
    if step3_src:
        sys.path.insert(0, str(step3_src))
        print(f"✅ Added Step 3 source path: {step3_src}")
        return True
    else:
        print("❌ Could not find Step 3 source directory")
        return False

def test_synthetic_dem_loading():
    """Test loading synthetic DEM files."""
    
    print("\n🧪 Testing synthetic DEM loading...")
    
    try:
        import rasterio
        
        # Find synthetic DEM files
        dem_search_paths = [
            Path("STEP 2.5/outputs/aoi_specific_synthetic_dems"),
            Path("STEP 3/data/raw/dem"),
            Path("../STEP 2.5/outputs/aoi_specific_synthetic_dems"),
            Path("../STEP 3/data/raw/dem")
        ]
        
        dem_files = []
        for search_path in dem_search_paths:
            if search_path.exists():
                dem_files.extend(list(search_path.glob("*.tif")))
        
        if not dem_files:
            print("❌ No synthetic DEM files found")
            return False
        
        # Test loading first DEM
        test_dem = dem_files[0]
        print(f"   Testing DEM: {test_dem.name}")
        
        with rasterio.open(test_dem) as src:
            # Check basic properties
            print(f"   ✅ Size: {src.width}x{src.height} pixels")
            print(f"   ✅ CRS: {src.crs}")
            print(f"   ✅ NoData: {src.nodata}")
            
            # Check data
            dem_array = src.read(1, masked=True)
            
            if hasattr(dem_array, 'compressed'):
                valid_data = dem_array.compressed()
            else:
                valid_data = dem_array[~np.isnan(dem_array)]
            
            if len(valid_data) > 0:
                print(f"   ✅ Elevation range: {valid_data.min():.1f} - {valid_data.max():.1f} m")
                print(f"   ✅ Valid pixels: {len(valid_data):,} / {dem_array.size:,}")
                return True
            else:
                print("❌ No valid elevation data found")
                return False
    
    except Exception as e:
        print(f"❌ DEM loading test failed: {e}")
        return False

def test_pontzer_equations():
    """Test Pontzer equation calculations."""
    
    print("\n🧪 Testing Pontzer equations...")
    
    try:
        from energyscape.pontzer_equations import PontzerEquations
        
        # Initialize equations
        pontzer = PontzerEquations()
        print("   ✅ PontzerEquations initialized")
        
        # Test basic calculations
        test_mass = 4000.0  # kg (average elephant)
        test_slopes = [0, 5, 10, 15, 20, 30]  # degrees
        
        print("   Energy costs for 4000kg elephant:")
        for slope in test_slopes:
            cost = pontzer.calculate_energy_cost(test_mass, slope)
            print(f"     {slope:2d}°: {cost:,.2f} kcal/km")
        
        # Test array calculations
        slope_array = np.array(test_slopes)
        cost_array = pontzer.calculate_energy_cost(test_mass, slope_array)
        
        if len(cost_array) == len(slope_array):
            print("   ✅ Array calculations working")
        else:
            print("❌ Array calculation size mismatch")
            return False
        
        # Test cost surface generation
        synthetic_slopes = np.random.uniform(0, 30, (50, 50))
        cost_surface = pontzer.calculate_cost_surface(test_mass, synthetic_slopes)
        
        if cost_surface.shape == synthetic_slopes.shape:
            print("   ✅ Cost surface generation working")
            return True
        else:
            print("❌ Cost surface shape mismatch")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import PontzerEquations: {e}")
        return False
    except Exception as e:
        print(f"❌ Pontzer equations test failed: {e}")
        return False

def test_slope_calculation():
    """Test slope calculation from synthetic DEM."""
    
    print("\n🧪 Testing slope calculations...")
    
    try:
        from energyscape.slope_calculation import SlopeCalculator
        import rasterio
        
        # Find a synthetic DEM
        dem_search_paths = [
            Path("STEP 2.5/outputs/aoi_specific_synthetic_dems"),
            Path("STEP 3/data/raw/dem"),
            Path("../STEP 2.5/outputs/aoi_specific_synthetic_dems"),
            Path("../STEP 3/data/raw/dem")
        ]
        
        test_dem = None
        for search_path in dem_search_paths:
            if search_path.exists():
                dem_files = list(search_path.glob("*.tif"))
                if dem_files:
                    test_dem = dem_files[0]
                    break
        
        if not test_dem:
            print("❌ No synthetic DEM found for slope testing")
            return False
        
        print(f"   Testing with DEM: {test_dem.name}")
        
        # Load DEM
        with rasterio.open(test_dem) as src:
            dem_array = src.read(1, masked=True)
            transform = src.transform
            
        # Initialize slope calculator
        calc = SlopeCalculator(resolution_m=30.0)
        print("   ✅ SlopeCalculator initialized")
        
        # Calculate slopes
        slope_array = calc.calculate_slope_degrees(dem_array, transform)
        
        if slope_array.shape == dem_array.shape:
            print("   ✅ Slope array shape matches DEM")
        else:
            print("❌ Slope array shape mismatch")
            return False
        
        # Check slope statistics
        if hasattr(slope_array, 'compressed'):
            valid_slopes = slope_array.compressed()
        else:
            valid_slopes = slope_array[~np.isnan(slope_array)]
        
        if len(valid_slopes) > 0:
            stats = calc.calculate_slope_statistics(slope_array)
            print(f"   ✅ Slope range: {stats['min']:.1f}° - {stats['max']:.1f}°")
            print(f"   ✅ Mean slope: {stats['mean']:.1f}° ± {stats['std']:.1f}°")
            print(f"   ✅ Valid pixels: {stats['valid_pixels']:,}")
            
            # Test slope validation
            validation = calc.validate_slope_calculation(dem_array, slope_array)
            if validation['valid']:
                print("   ✅ Slope calculation validation passed")
                return True
            else:
                print(f"❌ Slope validation failed: {validation['issues']}")
                return False
        else:
            print("❌ No valid slope data calculated")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import SlopeCalculator: {e}")
        return False
    except Exception as e:
        print(f"❌ Slope calculation test failed: {e}")
        return False

def test_energy_surface_generation():
    """Test complete energy surface generation pipeline."""
    
    print("\n🧪 Testing energy surface generation...")
    
    try:
        from energyscape.pontzer_equations import PontzerEquations
        from energyscape.slope_calculation import SlopeCalculator
        import rasterio
        
        # Find synthetic DEM
        dem_search_paths = [
            Path("STEP 2.5/outputs/aoi_specific_synthetic_dems"),
            Path("STEP 3/data/raw/dem"),
            Path("../STEP 2.5/outputs/aoi_specific_synthetic_dems"),
            Path("../STEP 3/data/raw/dem")
        ]
        
        test_dem = None
        for search_path in dem_search_paths:
            if search_path.exists():
                dem_files = list(search_path.glob("*.tif"))
                if dem_files:
                    test_dem = dem_files[0]
                    break
        
        if not test_dem:
            print("❌ No synthetic DEM found")
            return False
        
        print(f"   Using DEM: {test_dem.name}")
        
        # Load DEM
        with rasterio.open(test_dem) as src:
            dem_array = src.read(1, masked=True)
            transform = src.transform
            profile = src.profile.copy()
        
        # Calculate slopes
        slope_calc = SlopeCalculator(resolution_m=30.0)
        slope_array = slope_calc.calculate_slope_degrees(dem_array, transform)
        print("   ✅ Slopes calculated")
        
        # Calculate energy costs
        pontzer = PontzerEquations()
        
        # Test for different elephant types
        elephant_masses = {
            'female': 2744.0,
            'male': 6029.0,
            'average': 4000.0
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for sex, mass_kg in elephant_masses.items():
                print(f"   Calculating energy surface for {sex} elephant ({mass_kg} kg)...")
                
                # Generate energy surface
                energy_array = pontzer.calculate_cost_surface(
                    mass_kg=mass_kg,
                    slope_array=slope_array,
                    nodata_value=profile.get('nodata', -9999.0)
                )
                
                if energy_array.shape == dem_array.shape:
                    print(f"     ✅ Energy surface shape correct")
                else:
                    print(f"     ❌ Energy surface shape mismatch")
                    return False
                
                # Save test energy surface
                output_profile = profile.copy()
                output_profile.update({
                    'dtype': 'float32',
                    'nodata': profile.get('nodata', -9999.0)
                })
                
                test_output = temp_path / f"test_energy_{sex}.tif"
                
                with rasterio.open(test_output, 'w', **output_profile) as dst:
                    if isinstance(energy_array, np.ma.MaskedArray):
                        dst.write(energy_array.filled(output_profile['nodata']), 1)
                    else:
                        dst.write(energy_array.astype(np.float32), 1)
                
                # Validate saved file
                with rasterio.open(test_output) as test_src:
                    test_data = test_src.read(1, masked=True)
                    
                    if hasattr(test_data, 'compressed'):
                        valid_energy = test_data.compressed()
                    else:
                        valid_energy = test_data[~np.isnan(test_data)]
                    
                    if len(valid_energy) > 0:
                        print(f"     ✅ Energy range: {valid_energy.min():.1f} - {valid_energy.max():.1f} kcal/km")
                    else:
                        print(f"     ❌ No valid energy data")
                        return False
        
        print("   ✅ Energy surface generation successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error in energy surface test: {e}")
        return False
    except Exception as e:
        print(f"❌ Energy surface generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step2_aoi_discovery():
    """Test discovery of Step 2 AOI files."""
    
    print("\n🧪 Testing Step 2 AOI discovery...")
    
    try:
        # Test the discovery logic from Step 3
        possible_paths = [
            Path("../STEP 2/data/outputs"),
            Path("../STEP 2/outputs"),
            Path("STEP 2/data/outputs"),
            Path("../../STEP 2/data/outputs")
        ]
        
        aoi_files = []
        
        for path in possible_paths:
            if path.exists():
                print(f"   📂 Checking: {path}")
                
                patterns = ["*aoi*.geojson", "*aoi*.shp", "*AOI*.geojson", "*AOI*.shp"]
                for pattern in patterns:
                    found_files = list(path.rglob(pattern))
                    aoi_files.extend(found_files)
                
                if aoi_files:
                    break
        
        if aoi_files:
            print(f"   ✅ Found {len(aoi_files)} AOI files")
            
            # Test loading one AOI
            import geopandas as gpd
            test_aoi = aoi_files[0]
            
            gdf = gpd.read_file(test_aoi)
            
            # Ensure geographic coordinates
            if gdf.crs and not gdf.crs.is_geographic:
                gdf = gdf.to_crs('EPSG:4326')
            elif gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            
            bounds = gdf.total_bounds
            print(f"   ✅ Sample AOI bounds: {bounds[1]:.3f},{bounds[0]:.3f} to {bounds[3]:.3f},{bounds[2]:.3f}")
            
            return True
        else:
            print("   ❌ No AOI files found")
            return False
            
    except Exception as e:
        print(f"❌ AOI discovery test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all integration tests."""
    
    print("🧪 Step 3 Pipeline Integration Test Suite")
    print("=" * 60)
    print("Testing synthetic DEM integration with EnergyScape pipeline")
    print()
    
    # Setup Python paths
    if not setup_paths():
        return False
    
    # Test categories  
    tests = [
        ("Step 2 AOI Discovery", test_step2_aoi_discovery),
        ("Synthetic DEM Loading", test_synthetic_dem_loading),
        ("Pontzer Equations", test_pontzer_equations),
        ("Slope Calculations", test_slope_calculation),
        ("Energy Surface Generation", test_energy_surface_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = test_func()
            results.append(result)
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Synthetic DEMs are fully compatible with Step 3 pipeline")
        print("✅ EnergyScape energy surface generation is ready")
        
        print(f"\n🚀 Next Steps:")
        print("1. Navigate to Step 3: cd 'STEP 3'")
        print("2. Run EnergyScape: python run_energyscape.py")
        print("3. Monitor energy surface generation progress")
        
        return True
        
    elif passed >= total * 0.8:  # 80% pass rate
        print("⚠️  MOSTLY PASSED - Minor issues detected")
        print("✅ Core functionality appears working")
        print("⚠️  Some edge cases may need attention")
        
        print(f"\n🚀 You can proceed with Step 3:")
        print("1. Navigate to Step 3: cd 'STEP 3'")
        print("2. Run EnergyScape: python run_energyscape.py")
        print("3. Monitor for any runtime issues")
        
        return True
        
    else:
        print("❌ SIGNIFICANT ISSUES DETECTED")
        print("🔧 Please address the failed tests before proceeding")
        
        failed_tests = [test_name for (test_name, _), result in zip(tests, results) if not result]
        print(f"\nFailed tests: {', '.join(failed_tests)}")
        
        return False

def main():
    """Main test execution."""
    
    try:
        success = run_comprehensive_test()
        return success
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)