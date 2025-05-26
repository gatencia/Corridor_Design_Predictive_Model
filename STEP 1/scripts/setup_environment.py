# Script to verify environment setup
#!/usr/bin/env python3
"""
Environment Setup Verification Script
Verifies that all required libraries are correctly installed and functional.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class EnvironmentChecker:
    """Comprehensive environment verification for elephant corridor analysis."""
    
    def __init__(self):
        self.results = []
        self.critical_failures = []
        self.python_version = sys.version_info
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        print("🐍 Checking Python version...")
        
        if self.python_version >= (3, 9):
            print(f"   ✅ Python {sys.version.split()[0]} (>= 3.9 required)")
            return True
        else:
            print(f"   ❌ Python {sys.version.split()[0]} (>= 3.9 required)")
            self.critical_failures.append("Python version too old")
            return False
    
    def check_core_libraries(self) -> Dict[str, bool]:
        """Check core scientific computing libraries."""
        print("\n📊 Checking core scientific libraries...")
        
        core_libs = {
            'numpy': 'Numerical computing',
            'pandas': 'Data manipulation', 
            'scipy': 'Scientific computing',
            'matplotlib': 'Plotting'
        }
        
        results = {}
        for lib, description in core_libs.items():
            try:
                __import__(lib)
                print(f"   ✅ {lib:<12} - {description}")
                results[lib] = True
            except ImportError as e:
                print(f"   ❌ {lib:<12} - {description} (MISSING)")
                results[lib] = False
                self.critical_failures.append(f"Missing {lib}")
                
        return results
    
    def check_geospatial_libraries(self) -> Dict[str, bool]:
        """Check geospatial libraries and their dependencies."""
        print("\n🗺️  Checking geospatial libraries...")
        
        geo_libs = {
            'geopandas': 'Vector GIS operations',
            'rasterio': 'Raster I/O',
            'fiona': 'Vector file I/O',
            'shapely': 'Geometric operations',
            'pyproj': 'Coordinate transformations'
        }
        
        results = {}
        for lib, description in geo_libs.items():
            try:
                module = __import__(lib)
                version = getattr(module, '__version__', 'unknown')
                print(f"   ✅ {lib:<12} v{version:<8} - {description}")
                results[lib] = True
            except ImportError:
                print(f"   ❌ {lib:<12} - {description} (MISSING)")
                results[lib] = False
                self.critical_failures.append(f"Missing {lib}")
                
        return results
    
    def check_specialized_libraries(self) -> Dict[str, bool]:
        """Check specialized libraries for corridor analysis."""
        print("\n🔬 Checking specialized libraries...")
        
        specialized_libs = [
            ('scikit-image', 'Image processing & pathfinding'),
            ('folium', 'Interactive mapping'),
            ('contextily', 'Basemap tiles'),
            ('tqdm', 'Progress bars'),
            ('pytest', 'Testing framework')
        ]
        
        results = {}
        for lib, description in specialized_libs:
            try:
                module = __import__(lib.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                print(f"   ✅ {lib:<15} v{version:<8} - {description}")
                results[lib] = True
            except ImportError:
                print(f"   ❌ {lib:<15} - {description} (MISSING)")
                results[lib] = False
                
        return results
    
    def check_circuitscape(self) -> bool:
        """Check Circuitscape installation."""
        print("\n⚡ Checking Circuitscape...")
        
        try:
            import circuitscape
            print(f"   ✅ circuitscape v{circuitscape.__version__} - Circuit theory modeling")
            return True
        except ImportError:
            print("   ❌ circuitscape - Circuit theory modeling (MISSING)")
            print("      Install with: pip install circuitscape==4.0.5")
            return False
    
    def check_r_integration(self) -> bool:
        """Check R and rpy2 for enerscape integration."""
        print("\n📈 Checking R integration...")
        
        # Check rpy2
        try:
            import rpy2
            print(f"   ✅ rpy2 v{rpy2.__version__} - Python-R bridge")
            rpy2_ok = True
        except ImportError:
            print("   ❌ rpy2 - Python-R bridge (MISSING)")
            rpy2_ok = False
        
        # Check R installation
        try:
            result = subprocess.run(['R', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                r_version = result.stdout.split('\n')[0]
                print(f"   ✅ R installation found: {r_version}")
                r_ok = True
            else:
                print("   ❌ R installation not found")
                r_ok = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ❌ R installation not found or not in PATH")
            r_ok = False
        
        # Check enerscape R package
        if r_ok and rpy2_ok:
            try:
                from rpy2.robjects.packages import importr
                enerscape = importr('enerscape')
                print("   ✅ enerscape R package available")
                return True
            except Exception:
                print("   ⚠️  enerscape R package not found")
                print("      Install in R with: install.packages('enerscape')")
                return False
        
        return False
    
    def check_optional_libraries(self) -> Dict[str, bool]:
        """Check optional libraries for enhanced functionality."""
        print("\n🎯 Checking optional libraries...")
        
        optional_libs = [
            ('dash', 'Interactive dashboards'),
            ('plotly', 'Interactive plots'),
            ('jupyter', 'Jupyter notebooks'),
            ('black', 'Code formatting'),
            ('flake8', 'Code linting')
        ]
        
        results = {}
        for lib, description in optional_libs:
            try:
                __import__(lib)
                print(f"   ✅ {lib:<12} - {description}")
                results[lib] = True
            except ImportError:
                print(f"   ⚠️  {lib:<12} - {description} (optional)")
                results[lib] = False
                
        return results
    
    def check_system_dependencies(self) -> Dict[str, bool]:
        """Check system-level dependencies."""
        print("\n🔧 Checking system dependencies...")
        
        system_deps = ['gdal-config', 'geos-config', 'proj']
        results = {}
        
        for dep in system_deps:
            try:
                result = subprocess.run([dep, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    print(f"   ✅ {dep:<12} - {version}")
                    results[dep] = True
                else:
                    print(f"   ❌ {dep:<12} - Not found")
                    results[dep] = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print(f"   ❌ {dep:<12} - Not found or not in PATH")
                results[dep] = False
                
        return results
    
    def test_basic_functionality(self) -> bool:
        """Test basic functionality with sample operations."""
        print("\n🧪 Testing basic functionality...")
        
        try:
            # Test numpy
            import numpy as np
            arr = np.array([1, 2, 3])
            assert arr.sum() == 6
            print("   ✅ NumPy basic operations")
            
            # Test pandas
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            assert len(df) == 2
            print("   ✅ Pandas DataFrame operations")
            
            # Test matplotlib
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            plt.close(fig)
            print("   ✅ Matplotlib plotting")
            
            # Test geopandas if available
            try:
                import geopandas as gpd
                from shapely.geometry import Point
                
                # Create simple GeoDataFrame
                points = [Point(0, 0), Point(1, 1)]
                gdf = gpd.GeoDataFrame({'id': [1, 2]}, geometry=points)
                assert len(gdf) == 2
                print("   ✅ GeoPandas GeoDataFrame operations")
            except ImportError:
                print("   ⚠️  GeoPandas test skipped (not installed)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Basic functionality test failed: {e}")
            return False
    
    def check_project_structure(self) -> bool:
        """Check if project directory structure exists."""
        print("\n📁 Checking project structure...")
        
        expected_dirs = [
            'data/raw',
            'data/processed', 
            'data/outputs',
            'scripts',
            'config',
            'tests'
        ]
        
        project_root = Path(__file__).parent.parent
        all_exist = True
        
        for dir_path in expected_dirs:
            full_path = project_root / dir_path
            if full_path.exists():
                print(f"   ✅ {dir_path}")
            else:
                print(f"   ⚠️  {dir_path} (will be created when needed)")
                # Don't mark as failure since directories can be created on demand
        
        return True
    
    def generate_report(self) -> None:
        """Generate final environment report."""
        print("\n" + "="*60)
        print("📋 ENVIRONMENT SETUP REPORT")
        print("="*60)
        
        if not self.critical_failures:
            print("🎉 Environment setup is READY for elephant corridor analysis!")
            print("\n✅ All critical dependencies are installed and functional.")
            
            print(f"\n🐍 Python Version: {sys.version.split()[0]}")
            print(f"📍 Python Path: {sys.executable}")
            print(f"📦 Project Root: {Path(__file__).parent.parent.absolute()}")
            
        else:
            print("❌ Environment setup has CRITICAL ISSUES!")
            print("\n🚨 Critical failures that must be resolved:")
            for failure in self.critical_failures:
                print(f"   • {failure}")
            
            print("\n🔧 Suggested fixes:")
            print("1. Install missing packages with conda:")
            print("   conda env create -f environment.yml")
            print("2. Or install with pip:")
            print("   pip install -r requirements.txt")
            print("3. Re-run this script to verify fixes")
        
        print("\n" + "="*60)
    
    def run_all_checks(self) -> bool:
        """Run all environment checks."""
        print("🔍 ELEPHANT CORRIDOR ENVIRONMENT VERIFICATION")
        print("="*60)
        
        checks = [
            self.check_python_version(),
            bool(self.check_core_libraries()),
            bool(self.check_geospatial_libraries()),
            bool(self.check_specialized_libraries()),
            self.check_circuitscape(),
            self.check_r_integration(),
            bool(self.check_optional_libraries()),
            bool(self.check_system_dependencies()),
            self.test_basic_functionality(),
            self.check_project_structure()
        ]
        
        self.generate_report()
        
        return len(self.critical_failures) == 0

def main():
    """Main function to run environment verification."""
    checker = EnvironmentChecker()
    success = checker.run_all_checks()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()