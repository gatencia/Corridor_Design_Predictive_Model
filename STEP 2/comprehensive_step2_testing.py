#!/usr/bin/env python3
"""
Comprehensive Step 2 Testing Suite
Extensive testing to ensure GPS data processing is ready for Step 3 (EnergyScape).
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
import shutil
import glob # Or use Path.glob for more modern path operations
warnings.filterwarnings('ignore')

# Add src to Python path
current_script_dir = Path(__file__).parent.resolve() # Should be STEP 2 directory
step2_src_dir = current_script_dir / "src"
sys.path.insert(0, str(step2_src_dir))

# Add STEP 1 to Python path for ProjectConfig
step1_project_dir = current_script_dir.parent / "STEP 1"
sys.path.insert(0, str(step1_project_dir))

from data_ingestion import GPSDataProcessor

class Step2ComprehensiveTester:
    """Comprehensive testing for Step 2 GPS processing pipeline."""
    
    def __init__(self, project_root, data_dir_override=None, config_file_override=None): # Renamed config_file to config_file_override
        self.project_root = Path(project_root).resolve() # This is STEP 2 dir
        self.data_dir_override = data_dir_override
        # self.config_file = config_file # Original line
        
        self.test_results = {}
        self.errors = []
        self.warnings = []
        self.critical_failures = []
        
        # Directories
        self.reports_dir = self.project_root / "reports"
        self.test_exports_base_dir = self.project_root / "outputs"
        self.test_specific_export_dir = self.test_exports_base_dir / "comprehensive_test_files" # Dedicated folder for this script's exports

        self._cleanup_previous_outputs()

        # Ensure directories exist for the current run after cleanup
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.test_specific_export_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ProjectConfig and GPSDataProcessor
        try:
            from config.project_config import ProjectConfig # Should now import from STEP 1/config
            
            # Determine path to a potential main_config.yaml in STEP 1/config
            # project_root for ProjectConfig is STEP 1
            step1_config_dir = step1_project_dir / "config"
            default_step1_yaml_config = step1_config_dir / "main_config.yaml"

            actual_config_file_to_load = None
            if config_file_override: # If a specific config YAML is passed to ComprehensiveTester
                actual_config_file_to_load = config_file_override
            elif default_step1_yaml_config.exists():
                actual_config_file_to_load = str(default_step1_yaml_config)

            if actual_config_file_to_load:
                self.config = ProjectConfig(config_file=actual_config_file_to_load)
                print(f"INFO: ProjectConfig loaded from YAML: {actual_config_file_to_load}")
            else:
                self.config = ProjectConfig() # Initialize from STEP 1 defaults (project_config.py)
                print("INFO: ProjectConfig loaded with default settings from project_config.py.")
            
            print(f"INFO: ProjectConfig.paths.project_root is: {self.config.paths.project_root}")

        except ImportError as e:
            print(f"ERROR: Could not import ProjectConfig: {e}")
            print("       Ensure STEP 1/config/project_config.py exists and STEP 1 is in PYTHONPATH.")
            self.config = None
        except Exception as e:
            print(f"ERROR: Error initializing ProjectConfig: {e}")
            # import traceback
            # print(traceback.format_exc())
            self.config = None

        self.processor = GPSDataProcessor(config=self.config) # Pass config to processor

    def _cleanup_previous_outputs(self):
        print("\\n🧹 Cleaning up previous test outputs...")

        # 1. Clean old JSON reports from this script
        if self.reports_dir.exists():
            # Using Path.glob for cleaner file iteration
            for report_file in self.reports_dir.glob("step2_comprehensive_test_report_*.json"):
                try:
                    report_file.unlink()
                    print(f"   🗑️ Deleted old report: {report_file.name}")
                except OSError as e:
                    print(f"   ⚠️ Error deleting report {report_file}: {e}")
        
        # 2. Clean dedicated test export directory used by this script
        if self.test_specific_export_dir.exists():
            try:
                shutil.rmtree(self.test_specific_export_dir)
                print(f"   🗑️ Deleted old test export directory: {self.test_specific_export_dir}")
            except OSError as e:
                print(f"   ⚠️ Error deleting directory {self.test_specific_export_dir}: {e}")
        
        print("🧹 Cleanup complete.")
        # The directory self.test_specific_export_dir will be recreated in __init__

    def run_all_tests(self, data_dir):
        """Run complete test suite."""
        
        print("🧪 COMPREHENSIVE STEP 2 TESTING SUITE")
        print("=" * 80)
        print("Testing GPS data processing pipeline for Step 3 readiness...")
        
        # Test categories
        test_categories = [
            ("🔧 Environment & Dependencies", self.test_environment),
            ("📁 Data File Integrity", lambda: self.test_data_files(data_dir)),
            ("📊 GPS Data Quality", lambda: self.test_gps_data_quality(data_dir)),
            ("🗺️ Geospatial Processing", lambda: self.test_geospatial_processing(data_dir)),
            ("📐 AOI Generation", lambda: self.test_aoi_generation(data_dir)),
            ("💾 File Export & Format", lambda: self.test_file_exports(data_dir)),
            ("🔗 Step 3 Readiness", lambda: self.test_step3_readiness(data_dir)),
            ("⚡ Performance & Scale", lambda: self.test_performance(data_dir)),
            ("🛡️ Error Handling", lambda: self.test_error_handling(data_dir)),
            ("📈 Data Completeness", lambda: self.test_data_completeness(data_dir))
        ]
        
        # Run all test categories
        for category_name, test_function in test_categories:
            print(f"\n{category_name}")
            print("-" * 60)
            
            try:
                test_function()
            except Exception as e:
                self.critical_failures.append(f"{category_name}: {e}")
                print(f"❌ CRITICAL FAILURE in {category_name}: {e}")
        
        # Generate final report
        self.generate_final_report()
    
    def test_environment(self):
        """Test environment and dependencies."""
        
        print("Testing Python environment and required libraries...")
        
        # Test core libraries
        required_libs = [
            ('pandas', '2.0.0'),
            ('geopandas', '0.13.0'),
            ('numpy', '1.24.0'),
            ('shapely', '2.0.0'),
            ('pyproj', '3.6.0'),
            ('geopy', '2.3.0'),
            ('tqdm', '4.65.0')
        ]
        
        missing_libs = []
        version_issues = []
        
        for lib_name, min_version in required_libs:
            try:
                lib = __import__(lib_name)
                current_version = getattr(lib, '__version__', 'unknown')
                
                if current_version != 'unknown':
                    # Simple version comparison (major.minor)
                    current_parts = current_version.split('.')[:2]
                    min_parts = min_version.split('.')[:2]
                    
                    if [int(x) for x in current_parts] < [int(x) for x in min_parts]:
                        version_issues.append(f"{lib_name}: {current_version} < {min_version}")
                
                print(f"   ✅ {lib_name} v{current_version}")
                
            except ImportError:
                missing_libs.append(lib_name)
                print(f"   ❌ {lib_name} (missing)")
        
        # Test optional libraries
        optional_libs = ['contextily', 'matplotlib']
        for lib in optional_libs:
            try:
                __import__(lib)
                print(f"   ✅ {lib} (optional)")
            except ImportError:
                print(f"   ⚠️  {lib} (optional, missing)")
        
        if missing_libs:
            self.critical_failures.append(f"Missing required libraries: {missing_libs}")
        if version_issues:
            self.warnings.extend(version_issues)
        
        self.test_results['environment'] = {
            'missing_libraries': missing_libs,
            'version_issues': version_issues,
            'status': 'PASS' if not missing_libs else 'FAIL'
        }
    
    def test_data_files(self, data_dir):
        """Test data file integrity and format."""
        
        print(f"Testing GPS data files in {data_dir}...")
        
        if not data_dir.exists():
            self.critical_failures.append(f"Data directory not found: {data_dir}")
            return
        
        csv_files = list(data_dir.glob("*.csv"))
        print(f"   Found {len(csv_files)} CSV files")
        
        if len(csv_files) == 0:
            self.critical_failures.append("No CSV files found in data directory")
            return
        
        file_issues = []
        valid_files = 0
        total_rows = 0
        
        for csv_file in csv_files:
            try:
                # Basic file checks
                if csv_file.stat().st_size == 0:
                    file_issues.append(f"{csv_file.name}: Empty file")
                    continue
                
                # Try to read CSV
                df = pd.read_csv(csv_file, nrows=5)  # Just read first 5 rows
                
                # Check required columns
                required_cols = ['timestamp', 'location-lat', 'location-long', 
                               'tag-local-identifier', 'individual-local-identifier']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    file_issues.append(f"{csv_file.name}: Missing columns {missing_cols}")
                    continue
                
                # Count total rows (full file)
                full_df = pd.read_csv(csv_file)
                total_rows += len(full_df)
                valid_files += 1
                
                print(f"   ✅ {csv_file.name}: {len(full_df)} rows")
                
            except Exception as e:
                file_issues.append(f"{csv_file.name}: {str(e)}")
                print(f"   ❌ {csv_file.name}: {e}")
        
        print(f"   📊 Summary: {valid_files}/{len(csv_files)} valid files, {total_rows:,} total GPS points")
        
        if len(file_issues) > 0:
            self.warnings.extend(file_issues)
        
        self.test_results['data_files'] = {
            'total_files': len(csv_files),
            'valid_files': valid_files,
            'total_gps_points': total_rows,
            'file_issues': file_issues,
            'status': 'PASS' if valid_files > 0 else 'FAIL'
        }
    
    def test_gps_data_quality(self, data_dir):
        """Test GPS data quality and validation."""
        
        print("Testing GPS data quality and validation...")
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            return
        
        # Test with a sample of files
        test_files = csv_files[:5]  # Test first 5 files
        processor = GPSDataProcessor()
        
        quality_results = {
            'coordinate_validity': [],
            'temporal_issues': [],
            'speed_issues': [],
            'duplicate_counts': [],
            'data_gaps': []
        }
        
        for csv_file in test_files:
            try:
                print(f"   Testing {csv_file.name}...")
                
                # Load and process data
                gdf = processor.load_gps_data(csv_file)
                
                # Check coordinate validity
                if 'coord_valid' in gdf.columns:
                    coord_validity = gdf['coord_valid'].mean()
                    quality_results['coordinate_validity'].append(coord_validity)
                    print(f"      Coordinate validity: {coord_validity*100:.1f}%")
                
                # Check temporal ordering
                timestamps = pd.to_datetime(gdf['timestamp'])
                temporal_issues = (timestamps.diff() < pd.Timedelta(0)).sum()
                quality_results['temporal_issues'].append(temporal_issues)
                if temporal_issues > 0:
                    print(f"      ⚠️  {temporal_issues} temporal ordering issues")
                
                # Check movement speeds
                if 'speed_kmh' in gdf.columns:
                    speeds = gdf['speed_kmh'].dropna()
                    if len(speeds) > 0:
                        high_speed_count = (speeds > 100).sum()
                        quality_results['speed_issues'].append(high_speed_count)
                        if high_speed_count > 0:
                            print(f"      ⚠️  {high_speed_count} unrealistic speed readings")
                
                # Check duplicates
                if 'is_duplicate' in gdf.columns:
                    duplicate_count = gdf['is_duplicate'].sum()
                    quality_results['duplicate_counts'].append(duplicate_count)
                    if duplicate_count > 0:
                        print(f"      ⚠️  {duplicate_count} duplicate GPS fixes")
                
                # Check for large time gaps
                time_diffs = timestamps.diff().dt.total_seconds() / 3600  # hours
                large_gaps = (time_diffs > 48).sum()  # gaps > 48 hours
                quality_results['data_gaps'].append(large_gaps)
                if large_gaps > 0:
                    print(f"      ⚠️  {large_gaps} large time gaps (>48h)")
                
                print(f"      ✅ Quality check complete")
                
            except Exception as e:
                print(f"      ❌ Quality test failed: {e}")
                self.errors.append(f"Quality test failed for {csv_file.name}: {e}")
        
        # Calculate summary statistics
        avg_coord_validity = np.mean(quality_results['coordinate_validity']) if quality_results['coordinate_validity'] else 0
        total_temporal_issues = sum(quality_results['temporal_issues'])
        total_speed_issues = sum(quality_results['speed_issues'])
        total_duplicates = sum(quality_results['duplicate_counts'])
        total_gaps = sum(quality_results['data_gaps'])
        
        print(f"   📊 Quality Summary:")
        print(f"      Average coordinate validity: {avg_coord_validity*100:.1f}%")
        print(f"      Total temporal issues: {total_temporal_issues}")
        print(f"      Total speed issues: {total_speed_issues}")
        print(f"      Total duplicates: {total_duplicates}")
        print(f"      Total large gaps: {total_gaps}")
        
        # Quality thresholds for Step 3 readiness
        quality_pass = (
            avg_coord_validity > 0.95 and  # >95% valid coordinates
            total_speed_issues < len(test_files) * 10  # <10 speed issues per file on average
        )
        
        self.test_results['data_quality'] = {
            'avg_coordinate_validity': avg_coord_validity,
            'total_temporal_issues': total_temporal_issues,
            'total_speed_issues': total_speed_issues,
            'total_duplicates': total_duplicates,
            'total_gaps': total_gaps,
            'files_tested': len(test_files),
            'status': 'PASS' if quality_pass else 'WARN'
        }
    
    def test_geospatial_processing(self, data_dir):
        """Test geospatial processing capabilities."""
        
        print("Testing geospatial processing...")
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            return
        
        processor = GPSDataProcessor()
        test_file = csv_files[0]  # Use first file for testing
        
        try:
            print(f"   Testing with {test_file.name}...")
            
            # Load GPS data
            gdf = processor.load_gps_data(test_file)
            print(f"      ✅ Loaded {len(gdf)} GPS points")
            
            # Test CRS handling
            original_crs = gdf.crs
            print(f"      Original CRS: {original_crs}")
            
            # Test CRS transformations
            utm_gdf = gdf.to_crs('EPSG:32633')  # UTM 33N
            print(f"      ✅ UTM transformation successful")
            
            # Test coordinate bounds
            bounds = gdf.total_bounds
            print(f"      Spatial bounds: {bounds[1]:.3f}°-{bounds[3]:.3f}°N, {bounds[0]:.3f}°-{bounds[2]:.3f}°E")
            
            # Check if bounds are reasonable for Central Africa
            reasonable_bounds = (
                1 <= bounds[1] <= 15 and   # Latitude range
                8 <= bounds[0] <= 17 and   # Longitude range
                bounds[3] > bounds[1] and  # Max > Min lat
                bounds[2] > bounds[0]      # Max > Min lon
            )
            
            if not reasonable_bounds:
                self.warnings.append(f"GPS coordinates outside expected Central Africa bounds")
            
            # Test geometric operations
            convex_hull = gdf.unary_union.convex_hull
            hull_area_deg2 = convex_hull.area
            print(f"      ✅ Convex hull created: {hull_area_deg2:.6f} deg²")
            
            self.test_results['geospatial'] = {
                'crs_original': str(original_crs),
                'crs_transform_success': True,
                'bounds': bounds.tolist(),
                'bounds_reasonable': reasonable_bounds,
                'convex_hull_area': hull_area_deg2,
                'status': 'PASS'
            }
            
        except Exception as e:
            print(f"      ❌ Geospatial processing failed: {e}")
            self.errors.append(f"Geospatial processing error: {e}")
            self.test_results['geospatial'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_aoi_generation(self, data_dir):
        """Test AOI (Area of Interest) generation."""
        
        print("Testing AOI generation...")
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            return
        
        processor = GPSDataProcessor()
        aoi_results = []
        
        # Test AOI generation with multiple files
        test_files = csv_files[:3]  # Test first 3 files
        
        for csv_file in test_files:
            try:
                print(f"   Testing AOI for {csv_file.name}...")
                
                gdf = processor.load_gps_data(csv_file)
                
                if len(gdf) < 3:
                    print(f"      ⚠️  Insufficient points ({len(gdf)}) for AOI")
                    continue
                
                # Test different buffer sizes
                for buffer_km in [1.0, 5.0, 10.0]:
                    aoi_gdf = processor.generate_aoi(gdf, buffer_km=buffer_km)
                    
                    aoi_area = aoi_gdf['area_km2'].iloc[0]
                    
                    # AOI should be larger than GPS point spread
                    gps_bounds = gdf.total_bounds
                    gps_span_km = (
                        geodesic((gps_bounds[1], gps_bounds[0]), 
                               (gps_bounds[3], gps_bounds[2])).kilometers
                    )
                    
                    aoi_reasonable = aoi_area > (gps_span_km ** 2) / 4  # Rough check
                    
                    aoi_results.append({
                        'file': csv_file.name,
                        'buffer_km': buffer_km,
                        'area_km2': aoi_area,
                        'reasonable': aoi_reasonable
                    })
                    
                    print(f"      ✅ {buffer_km}km buffer: {aoi_area:.1f} km²")
                
            except Exception as e:
                print(f"      ❌ AOI generation failed: {e}")
                self.errors.append(f"AOI generation error for {csv_file.name}: {e}")
        
        # Check AOI generation success rate
        successful_aois = len([r for r in aoi_results if r['reasonable']])
        total_aois = len(aoi_results)
        
        aoi_success_rate = successful_aois / total_aois if total_aois > 0 else 0
        
        print(f"   📊 AOI Summary: {successful_aois}/{total_aois} reasonable AOIs ({aoi_success_rate*100:.1f}%)")
        
        self.test_results['aoi_generation'] = {
            'total_aois_tested': total_aois,
            'successful_aois': successful_aois,
            'success_rate': aoi_success_rate,
            'aoi_details': aoi_results,
            'status': 'PASS' if aoi_success_rate > 0.8 else 'WARN'
        }
    
    def test_file_exports(self, data_dir):
        """Test file export capabilities."""
        
        print("Testing file export formats...")
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            return
        
        processor = GPSDataProcessor()
        test_file = csv_files[0]
        
        try:
            # Create test output directory
            test_output_dir = Path("test_outputs")
            test_output_dir.mkdir(exist_ok=True)
            
            # Load test data
            gdf = processor.load_gps_data(test_file)
            aoi_gdf = processor.generate_aoi(gdf, buffer_km=5.0)
            
            print(f"   Testing exports with {test_file.name}...")
            base_filename = f"test_export_{test_file.stem}"
            output_formats = ['GeoParquet', 'CSV', 'GeoJSON', 'Shapefile']
            export_results = {}

            # CRITICAL: Ensure processor.export_data saves into self.test_specific_export_dir
            # This might require processor.export_data to accept an output_dir argument,
            # or for the processor to be configured to use this path.
            # For this example, I'm assuming export_data can take an output_dir_override.
            # If not, GPSDataProcessor needs modification, or this test needs to know
            # the exact output filenames to clean them up individually from a shared directory.

            try:
                # Ideal scenario: export_data allows specifying the output directory
                actual_export_files = self.processor.export_data(
                    gdf, 
                    aoi_gdf, 
                    base_filename, 
                    output_formats,
                    output_dir_override=self.test_specific_export_dir # Ensure files go here
                )

                for fmt, file_info in actual_export_files.items():
                    file_path = Path(file_info['path']) # Assuming path is returned
                    if file_path.exists():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        print(f"      ✅ {fmt}: {size_mb:.2f} MB (saved in {self.test_specific_export_dir.name})")
                        export_results[fmt] = True
                    else:
                        print(f"      ❌ {fmt}: File not found at {file_path}")
                        export_results[fmt] = False
                
                self.test_results['file_exports'].append(all(export_results.values()))

            except Exception as e:
                print(f"      ❌ Error during file export test: {e}")
                # import traceback
                # print(traceback.format_exc())
                self.test_results['file_exports'].append(False)
            
            # Clean up test files
            import shutil
            shutil.rmtree(test_output_dir, ignore_errors=True)
            
            successful_exports = sum(1 for r in export_results.values() if r['success'])
            total_exports = len(export_results)
            
            print(f"   📊 Export Summary: {successful_exports}/{total_exports} formats successful")
            
            self.test_results['file_exports'] = {
                'export_results': export_results,
                'successful_exports': successful_exports,
                'total_exports': total_exports,
                'status': 'PASS' if successful_exports >= 3 else 'WARN'  # Need at least 3 formats
            }
            
        except Exception as e:
            print(f"   ❌ Export testing failed: {e}")
            self.errors.append(f"Export testing error: {e}")
            self.test_results['file_exports'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_step3_readiness(self, data_dir):
        """Test readiness for Step 3 (EnergyScape) requirements."""
        
        print("Testing Step 3 (EnergyScape) readiness...")
        
        step3_requirements = {
            'projected_crs': False,
            'utm_coordinates': False,
            'aoi_polygons': False,
            'gps_points_valid': False,
            'spatial_extent_reasonable': False,
            'file_formats_compatible': False
        }
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            return
        
        processor = GPSDataProcessor()
        test_file = csv_files[0]
        
        try:
            # Load test data
            gdf = processor.load_gps_data(test_file)
            aoi_gdf = processor.generate_aoi(gdf, buffer_km=5.0)
            
            print(f"   Checking Step 3 requirements with {test_file.name}...")
            
            # 1. Check projected CRS capability
            try:
                utm_gdf = gdf.to_crs('EPSG:32633')  # UTM 33N
                step3_requirements['projected_crs'] = True
                step3_requirements['utm_coordinates'] = True
                print(f"      ✅ Projected CRS transformation: UTM coordinates available")
            except:
                print(f"      ❌ Projected CRS transformation failed")
            
            # 2. Check AOI polygon availability
            if len(aoi_gdf) > 0 and aoi_gdf.geometry.iloc[0].is_valid:
                step3_requirements['aoi_polygons'] = True
                print(f"      ✅ AOI polygons: Valid study area boundaries")
            else:
                print(f"      ❌ AOI polygons: Invalid or missing")
            
            # 3. Check GPS point validity for energy calculations
            valid_gps_ratio = gdf['coord_valid'].mean() if 'coord_valid' in gdf.columns else 1.0
            if valid_gps_ratio > 0.95:  # >95% valid points
                step3_requirements['gps_points_valid'] = True
                print(f"      ✅ GPS point validity: {valid_gps_ratio*100:.1f}% valid")
            else:
                print(f"      ❌ GPS point validity: {valid_gps_ratio*100:.1f}% valid (need >95%)")
            
            # 4. Check spatial extent for DEM requirements
            bounds = gdf.total_bounds
            lat_span = bounds[3] - bounds[1]
            lon_span = bounds[2] - bounds[0]
            
            # Reasonable extent for DEM processing (not too small, not too large)
            if 0.01 < lat_span < 10 and 0.01 < lon_span < 10:  # 1-1000 km roughly
                step3_requirements['spatial_extent_reasonable'] = True
                print(f"      ✅ Spatial extent: {lat_span:.3f}° × {lon_span:.3f}° (suitable for DEM)")
            else:
                print(f"      ❌ Spatial extent: {lat_span:.3f}° × {lon_span:.3f}° (may cause DEM issues)")
            
            # 5. Check file format compatibility
            try:
                # Test that we can export in formats needed for Step 3
                test_dir = Path("temp_step3_test")
                test_dir.mkdir(exist_ok=True)
                
                # GeoJSON (for web GIS)
                aoi_gdf.to_file(test_dir / "test.geojson", driver='GeoJSON')
                
                # Shapefile (for desktop GIS)
                aoi_gdf.to_file(test_dir / "test.shp")
                
                # CSV with coordinates (for R integration)
                df_export = gdf.drop(columns=['geometry']).copy()
                df_export['longitude'] = gdf.geometry.x
                df_export['latitude'] = gdf.geometry.y
                df_export.to_csv(test_dir / "test.csv", index=False)
                
                step3_requirements['file_formats_compatible'] = True
                print(f"      ✅ File formats: Compatible with Step 3 requirements")
                
                # Clean up
                import shutil
                shutil.rmtree(test_dir, ignore_errors=True)
                
            except Exception as e:
                print(f"      ❌ File formats: Compatibility issues - {e}")
            
        except Exception as e:
            print(f"   ❌ Step 3 readiness test failed: {e}")
            self.errors.append(f"Step 3 readiness error: {e}")
        
        # Calculate overall Step 3 readiness
        requirements_met = sum(step3_requirements.values())
        total_requirements = len(step3_requirements)
        readiness_score = requirements_met / total_requirements
        
        print(f"   📊 Step 3 Readiness: {requirements_met}/{total_requirements} requirements met ({readiness_score*100:.1f}%)")
        
        # Define readiness levels
        if readiness_score >= 0.9:
            readiness_status = "READY"
            readiness_message = "✅ Fully ready for Step 3 (EnergyScape)"
        elif readiness_score >= 0.7:
            readiness_status = "MOSTLY_READY"
            readiness_message = "⚠️  Mostly ready for Step 3 - minor issues to address"
        else:
            readiness_status = "NOT_READY"
            readiness_message = "❌ Not ready for Step 3 - significant issues need resolution"
        
        print(f"   🎯 {readiness_message}")
        
        self.test_results['step3_readiness'] = {
            'requirements': step3_requirements,
            'requirements_met': requirements_met,
            'total_requirements': total_requirements,
            'readiness_score': readiness_score,
            'readiness_status': readiness_status,
            'readiness_message': readiness_message,
            'status': 'PASS' if readiness_score >= 0.8 else 'WARN'
        }
    
    def test_performance(self, data_dir):
        """Test performance and scalability."""
        
        print("Testing performance and scalability...")
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            return
        
        processor = GPSDataProcessor()
        
        # Test with different file sizes
        file_sizes = []
        processing_times = []
        
        for csv_file in csv_files[:5]:  # Test first 5 files
            try:
                file_size_mb = csv_file.stat().st_size / (1024 * 1024)
                
                start_time = datetime.now()
                gdf = processor.load_gps_data(csv_file)
                aoi_gdf = processor.generate_aoi(gdf, buffer_km=5.0)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                
                file_sizes.append(file_size_mb)
                processing_times.append(processing_time)
                
                print(f"   {csv_file.name}: {file_size_mb:.2f} MB processed in {processing_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Performance test failed for {csv_file.name}: {e}")
        
        if processing_times:
            avg_processing_time = np.mean(processing_times)
            max_processing_time = max(processing_times)
            
            # Performance thresholds
            performance_acceptable = (
                avg_processing_time < 30 and  # Average < 30 seconds
                max_processing_time < 120     # Max < 2 minutes
            )
            
            print(f"   📊 Performance Summary:")
            print(f"      Average processing time: {avg_processing_time:.2f}s")
            print(f"      Maximum processing time: {max_processing_time:.2f}s")
            print(f"      Performance: {'✅ Acceptable' if performance_acceptable else '⚠️  Slow'}")
            
            self.test_results['performance'] = {
                'avg_processing_time': avg_processing_time,
                'max_processing_time': max_processing_time,
                'files_tested': len(processing_times),
                'performance_acceptable': performance_acceptable,
                'status': 'PASS' if performance_acceptable else 'WARN'
            }
        else:
            self.test_results['performance'] = {'status': 'FAIL', 'error': 'No performance data collected'}
    
    def test_error_handling(self, data_dir):
        """Test error handling and robustness."""
        
        print("Testing error handling and robustness...")
        
        processor = GPSDataProcessor()
        
        error_handling_tests = {
            'empty_file': False,
            'corrupted_data': False,
            'missing_columns': False,
            'invalid_coordinates': False,
            'invalid_timestamps': False
        }
        
        try:
            # Create test directory
            test_dir = Path("error_test_data")
            test_dir.mkdir(exist_ok=True)
            
            # Test 1: Empty file
            try:
                empty_file = test_dir / "empty.csv"
                empty_file.write_text("")
                processor.load_gps_data(empty_file)
                print(f"      ❌ Empty file: No error raised")
            except Exception:
                error_handling_tests['empty_file'] = True
                print(f"      ✅ Empty file: Error properly handled")
            
            # Test 2: Missing required columns
            try:
                missing_cols_file = test_dir / "missing_cols.csv"
                missing_cols_file.write_text("col1,col2\n1,2\n3,4\n")
                processor.load_gps_data(missing_cols_file)
                print(f"      ❌ Missing columns: No error raised")
            except Exception:
                error_handling_tests['missing_columns'] = True
                print(f"      ✅ Missing columns: Error properly handled")
            
            # Test 3: Invalid coordinates
            try:
                invalid_coords_file = test_dir / "invalid_coords.csv"
                csv_content = """timestamp,location-lat,location-long,tag-local-identifier,individual-local-identifier
2024-01-01 12:00:00,999,999,TEST001,1
2024-01-01 13:00:00,-999,-999,TEST001,1"""
                invalid_coords_file.write_text(csv_content)
                gdf = processor.load_gps_data(invalid_coords_file)
                if 'coord_valid' in gdf.columns and not gdf['coord_valid'].all():
                    error_handling_tests['invalid_coordinates'] = True
                    print(f"      ✅ Invalid coordinates: Properly flagged")
                else:
                    print(f"      ❌ Invalid coordinates: Not detected")
            except Exception as e:
                print(f"      ⚠️  Invalid coordinates test failed: {e}")
            
            # Test 4: Invalid timestamps
            try:
                invalid_time_file = test_dir / "invalid_time.csv"
                csv_content = """timestamp,location-lat,location-long,tag-local-identifier,individual-local-identifier
invalid_date,8.0,13.0,TEST001,1
2024-01-01 13:00:00,8.1,13.1,TEST001,1"""
                invalid_time_file.write_text(csv_content)
                processor.load_gps_data(invalid_time_file)
                print(f"      ❌ Invalid timestamps: No error raised")
            except Exception:
                error_handling_tests['invalid_timestamps'] = True
                print(f"      ✅ Invalid timestamps: Error properly handled")
            
            # Clean up
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"   ❌ Error handling test setup failed: {e}")
        
        successful_error_tests = sum(error_handling_tests.values())
        total_error_tests = len(error_handling_tests)
        
        print(f"   📊 Error Handling: {successful_error_tests}/{total_error_tests} tests passed")
        
        self.test_results['error_handling'] = {
            'tests': error_handling_tests,
            'successful_tests': successful_error_tests,
            'total_tests': total_error_tests,
            'status': 'PASS' if successful_error_tests >= total_error_tests * 0.8 else 'WARN'
        }
    
    def test_data_completeness(self, data_dir):
        """Test data completeness for corridor analysis."""
        
        print("Testing data completeness for corridor analysis...")
        
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            return
        
        processor = GPSDataProcessor()
        
        completeness_metrics = {
            'study_sites': set(),
            'temporal_coverage': [],
            'spatial_coverage': [],
            'individual_counts': [],
            'gps_point_counts': []
        }
        
        for csv_file in csv_files:
            try:
                gdf = processor.load_gps_data(csv_file)
                
                # Extract study site
                filename_parts = csv_file.stem.split(" - ")
                if len(filename_parts) >= 2:
                    study_site = filename_parts[1].replace("(Cameroon)", "").replace("(Nigeria)", "").strip()
                    completeness_metrics['study_sites'].add(study_site)
                
                # Temporal coverage
                date_range = gdf['timestamp'].max() - gdf['timestamp'].min()
                completeness_metrics['temporal_coverage'].append(date_range.days)
                
                # Spatial coverage (convex hull area)
                bounds = gdf.total_bounds
                spatial_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])  # deg²
                completeness_metrics['spatial_coverage'].append(spatial_area)
                
                # Individual and GPS counts
                completeness_metrics['individual_counts'].append(gdf['individual-local-identifier'].nunique())
                completeness_metrics['gps_point_counts'].append(len(gdf))
                
            except Exception as e:
                print(f"   ⚠️  Could not analyze {csv_file.name}: {e}")
        
        # Analyze completeness
        num_study_sites = len(completeness_metrics['study_sites'])
        avg_temporal_coverage = np.mean(completeness_metrics['temporal_coverage'])
        total_gps_points = sum(completeness_metrics['gps_point_counts'])
        total_individuals = sum(completeness_metrics['individual_counts'])
        
        print(f"   📊 Data Completeness Summary:")
        print(f"      Study sites: {num_study_sites}")
        print(f"      Average tracking duration: {avg_temporal_coverage:.0f} days")
        print(f"      Total GPS points: {total_gps_points:,}")
        print(f"      Total individuals: {total_individuals}")
        
        # Study site coverage (good for corridor analysis)
        study_sites_adequate = num_study_sites >= 5  # Need multiple sites for corridors
        temporal_adequate = avg_temporal_coverage >= 30  # At least 30 days average
        gps_adequate = total_gps_points >= 1000  # At least 1000 points total
        
        completeness_score = sum([study_sites_adequate, temporal_adequate, gps_adequate]) / 3
        
        if completeness_score >= 0.8:
            completeness_status = "EXCELLENT"
        elif completeness_score >= 0.6:
            completeness_status = "GOOD"
        else:
            completeness_status = "INSUFFICIENT"
        
        print(f"   🎯 Data Completeness: {completeness_status} ({completeness_score*100:.0f}%)")
        
        self.test_results['data_completeness'] = {
            'num_study_sites': num_study_sites,
            'avg_temporal_coverage': avg_temporal_coverage,
            'total_gps_points': total_gps_points,
            'total_individuals': total_individuals,
            'completeness_score': completeness_score,
            'completeness_status': completeness_status,
            'study_sites': list(completeness_metrics['study_sites']),
            'status': 'PASS' if completeness_score >= 0.6 else 'WARN'
        }
    
    def generate_final_report(self):
        """Generate comprehensive final test report."""
        
        print(f"\n🎯 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        # Count test results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'PASS')
        warned_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'WARN')
        failed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'FAIL')
        
        print(f"Test Results: {passed_tests} PASS, {warned_tests} WARN, {failed_tests} FAIL")
        
        # Overall status
        if failed_tests > 0 or len(self.critical_failures) > 0:
            overall_status = "❌ CRITICAL ISSUES - NOT READY FOR STEP 3"
        elif warned_tests > total_tests / 2:
            overall_status = "⚠️  WARNINGS - PROCEED WITH CAUTION"
        else:
            overall_status = "✅ READY FOR STEP 3 (ENERGYSCAPE)"
        
        print(f"\nOverall Status: {overall_status}")
        
        # Critical failures
        if self.critical_failures:
            print(f"\n🚨 Critical Failures ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"   • {failure}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"   • {warning}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more warnings")
        
        # Key metrics for Step 3
        print(f"\n📊 Key Metrics for Step 3:")
        
        if 'step3_readiness' in self.test_results:
            readiness = self.test_results['step3_readiness']
            print(f"   Step 3 Readiness Score: {readiness['readiness_score']*100:.0f}%")
            print(f"   Status: {readiness['readiness_status']}")
        
        if 'data_completeness' in self.test_results:
            completeness = self.test_results['data_completeness']
            print(f"   Study Sites: {completeness['num_study_sites']}")
            print(f"   Total GPS Points: {completeness['total_gps_points']:,}")
            print(f"   Data Completeness: {completeness['completeness_status']}")
        
        if 'data_quality' in self.test_results:
            quality = self.test_results['data_quality']
            print(f"   Coordinate Validity: {quality['avg_coordinate_validity']*100:.1f}%")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        
        if failed_tests == 0 and len(self.critical_failures) == 0:
            print("   ✅ Your GPS data processing pipeline is ready for Step 3!")
            print("   ✅ All critical requirements are met for EnergyScape implementation")
            print("   ✅ Proceed with confidence to Step 3: Energy landscape calculations")
        else:
            print("   🔧 Address critical failures before proceeding to Step 3")
            print("   🔧 Review data quality issues and reprocess problematic files")
            if warned_tests > 0:
                print("   ⚠️  Monitor warning conditions during Step 3 processing")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path("reports") / f"step2_comprehensive_test_report_{timestamp}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        detailed_report = {
            'test_timestamp': timestamp,
            'overall_status': overall_status,
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'warned_tests': warned_tests,
                'failed_tests': failed_tests
            },
            'critical_failures': self.critical_failures,
            'warnings': self.warnings,
            'errors': self.errors,
            'detailed_results': self.test_results
        }
        
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        print(f"📄 File size: {report_file.stat().st_size / 1024:.1f} KB")

def main():
    """Run comprehensive Step 2 testing."""
    
    # Find GPS data directory
    data_paths = [
        Path("../GPS_Collar_CSV_Mark"), # Relative to STEP 2 directory
        Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/GPS_Collar_CSV_Mark") # Absolute path as fallback
    ]
    
    data_dir = None
    for path_option in data_paths:
        # If the path is relative, resolve it against the script's directory (current_script_dir)
        potential_data_dir = path_option if path_option.is_absolute() else current_script_dir.parent / path_option
        if potential_data_dir.exists() and potential_data_dir.is_dir():
            data_dir = potential_data_dir.resolve()
            print(f"INFO: Found data directory at: {data_dir}")
            break
    
    if not data_dir:
        print("❌ CRITICAL: Could not find GPS data directory. Checked:")
        for path_option in data_paths:
            potential_data_dir = path_option if path_option.is_absolute() else current_script_dir.parent / path_option
            print(f"    - {potential_data_dir.resolve()}")
        print("Please ensure GPS data is available in one of the expected locations for testing.")
        return
    
    # Run comprehensive testing
    # Use the globally defined current_script_dir which is the STEP 2 directory
    tester = Step2ComprehensiveTester(project_root=current_script_dir, data_dir_override=data_dir)
    tester.run_all_tests(data_dir) # data_dir is passed again for specific test functions that use it directly

if __name__ == "__main__":
    # Import required libraries for testing
    try:
        from geopy.distance import geodesic
    except ImportError:
        print("⚠️  geopy not available - some tests may be limited")
        geodesic = None
    
    main()