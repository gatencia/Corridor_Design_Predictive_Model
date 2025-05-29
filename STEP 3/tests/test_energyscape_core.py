#!/usr/bin/env python3
"""
Unit tests for EnergyScape core functionality.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

class TestPontzerEquations:
    """Test suite for Pontzer equation implementations."""
    
    def test_base_cost_calculation(self):
        """Test base energy cost calculation."""
        from energyscape.pontzer_equations import PontzerEquations
        
        pontzer = PontzerEquations()
        
        # Test with standard elephant mass
        mass_kg = 4000.0
        base_cost = pontzer.calculate_base_cost(mass_kg)
        
        assert base_cost > 0, "Base cost should be positive"
        assert isinstance(base_cost, (int, float)), "Base cost should be numeric"
        
        # Test scaling with mass
        smaller_cost = pontzer.calculate_base_cost(2000.0)
        larger_cost = pontzer.calculate_base_cost(6000.0)
        
        assert smaller_cost < base_cost < larger_cost, "Cost should scale with mass"
    
    def test_slope_multiplier(self):
        """Test slope cost multiplier calculation."""
        from energyscape.pontzer_equations import PontzerEquations
        
        pontzer = PontzerEquations()
        
        # Test flat terrain
        flat_multiplier = pontzer.calculate_slope_multiplier(0.0)
        assert flat_multiplier == 1.0, "Flat terrain should have multiplier of 1.0"
        
        # Test slope increases cost
        slope_multiplier = pontzer.calculate_slope_multiplier(15.0)
        assert slope_multiplier > 1.0, "Slope should increase cost"
        
        # Test steeper slope increases cost more
        steep_multiplier = pontzer.calculate_slope_multiplier(30.0)
        assert steep_multiplier > slope_multiplier, "Steeper slopes should cost more"
    
    def test_energy_cost_calculation(self):
        """Test total energy cost calculation."""
        from energyscape.pontzer_equations import PontzerEquations
        
        pontzer = PontzerEquations()
        
        # Test basic calculation
        mass_kg = 4000.0
        slope_degrees = 15.0
        
        cost = pontzer.calculate_energy_cost(mass_kg, slope_degrees)
        assert cost > 0, "Energy cost should be positive"
        
        # Test array calculation
        slopes = np.array([0, 5, 10, 15, 20])
        costs = pontzer.calculate_energy_cost(mass_kg, slopes)
        
        assert len(costs) == len(slopes), "Output should match input array length"
        assert np.all(costs > 0), "All costs should be positive"
        assert np.all(np.diff(costs) > 0), "Costs should increase with slope"
    
    def test_cost_surface_generation(self):
        """Test energy cost surface generation."""
        from energyscape.pontzer_equations import PontzerEquations
        
        pontzer = PontzerEquations()
        
        # Create test slope array
        slope_array = np.random.uniform(0, 30, (50, 50))
        mass_kg = 4000.0
        
        cost_surface = pontzer.calculate_cost_surface(mass_kg, slope_array)
        
        assert cost_surface.shape == slope_array.shape, "Output shape should match input"
        assert np.all(cost_surface[cost_surface != -9999] > 0), "Valid costs should be positive"

class TestSlopeCalculation:
    """Test suite for slope calculation functions."""
    
    def test_basic_slope_calculation(self):
        """Test basic slope calculation."""
        from energyscape.slope_calculation import SlopeCalculator
        
        calc = SlopeCalculator(resolution_m=30.0)
        
        # Create simple test DEM
        rows, cols = 10, 10
        dem = np.zeros((rows, cols))
        
        # Add gradient (1m per pixel)
        for i in range(rows):
            dem[i, :] = i
        
        slopes = calc.calculate_slope_degrees(dem)
        
        assert slopes.shape == dem.shape, "Output shape should match input"
        assert np.all(slopes >= 0), "Slopes should be non-negative"
    
    def test_slope_statistics(self):
        """Test slope statistics calculation."""
        from energyscape.slope_calculation import SlopeCalculator
        
        calc = SlopeCalculator()
        
        # Create test slope data
        slope_array = np.random.uniform(0, 45, (100, 100))
        
        stats = calc.calculate_slope_statistics(slope_array)
        
        required_keys = ['min', 'max', 'mean', 'median', 'std', 'valid_pixels']
        for key in required_keys:
            assert key in stats, f"Statistics should include {key}"
        
        assert 0 <= stats['min'] <= stats['max'] <= 45, "Statistics should be in expected range"
    
    def test_slope_validation(self):
        """Test slope calculation validation."""
        from energyscape.slope_calculation import SlopeCalculator
        
        calc = SlopeCalculator()
        
        # Create test data
        dem_array = np.random.uniform(0, 100, (50, 50))
        slope_array = np.random.uniform(0, 30, (50, 50))
        
        validation = calc.validate_slope_calculation(dem_array, slope_array)
        
        assert 'valid' in validation, "Validation should include valid flag"
        assert 'issues' in validation, "Validation should include issues list"
        assert isinstance(validation['valid'], bool), "Valid should be boolean"

class TestDEMLoader:
    """Test suite for DEM loading functionality."""
    
    def test_dem_info_creation(self):
        """Test DEM info object creation."""
        # This test would require actual DEM files
        # For now, just test that the class can be imported
        from dem_processing.dem_loader import DEMLoader, DEMInfo
        
        loader = DEMLoader()
        assert loader is not None, "DEMLoader should initialize"
    
    def test_find_dem_files(self):
        """Test DEM file discovery."""
        from dem_processing.dem_loader import DEMLoader
        
        loader = DEMLoader()
        
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test files
            (temp_path / "test.tif").touch()
            (temp_path / "test.img").touch()
            (temp_path / "not_dem.txt").touch()
            
            dem_files = loader.find_dem_files(temp_path)
            
            assert len(dem_files) == 2, "Should find 2 DEM files"
            assert all(f.suffix.lower() in ['.tif', '.img'] for f in dem_files), "Should only find DEM extensions"

class TestEnergyScapeConfig:
    """Test suite for EnergyScape configuration."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        from config.energyscape_config import EnergyScapeConfig
        
        config = EnergyScapeConfig()
        
        # Test default values
        assert config.pontzer_alpha == 1.0, "Default alpha should be 1.0"
        assert config.pontzer_beta == 0.75, "Default beta should be 0.75"
        assert config.pontzer_gamma == 1.5, "Default gamma should be 1.5"
        
        # Test elephant masses
        masses = config.get_elephant_masses()
        assert 'female' in masses, "Should include female mass"
        assert 'male' in masses, "Should include male mass"
        assert masses['male'] > masses['female'], "Male mass should be larger"
    
    def test_pontzer_parameters(self):
        """Test Pontzer parameter access."""
        from config.energyscape_config import EnergyScapeConfig
        
        config = EnergyScapeConfig()
        params = config.get_pontzer_parameters()
        
        required_params = ['alpha', 'beta', 'gamma']
        for param in required_params:
            assert param in params, f"Should include {param} parameter"
            assert params[param] > 0, f"{param} should be positive"

class TestRIntegration:
    """Test suite for R integration (if available)."""
    
    def test_r_availability_check(self):
        """Test R availability checking."""
        from energyscape.r_integration import EnerscapeRBridge
        
        bridge = EnerscapeRBridge()
        
        # Should not raise exception
        assert hasattr(bridge, 'r_available'), "Should have r_available attribute"
        assert isinstance(bridge.r_available, bool), "r_available should be boolean"
    
    def test_python_fallback(self):
        """Test Python fallback calculator."""
        from energyscape.r_integration import PythonEnergyCalculator
        
        calc = PythonEnergyCalculator()
        assert calc is not None, "Python calculator should initialize"

class TestEnergyScapeCore:
    """Test suite for EnergyScape core functionality."""
    
    def test_processor_initialization(self):
        """Test EnergyScape processor initialization."""
        from energyscape.core import EnergyScapeProcessor
        
        processor = EnergyScapeProcessor()
        assert processor is not None, "Processor should initialize"
        
        # Test components
        assert hasattr(processor, 'pontzer'), "Should have Pontzer equations"
        assert hasattr(processor, 'slope_calculator'), "Should have slope calculator"
        assert hasattr(processor, 'dem_loader'), "Should have DEM loader"
    
    def test_step2_aoi_discovery(self):
        """Test Step 2 AOI discovery."""
        from energyscape.core import EnergyScapeProcessor
        
        processor = EnergyScapeProcessor()
        
        # This will likely return empty list without actual Step 2 outputs
        aoi_files = processor.process_step2_aoi()
        assert isinstance(aoi_files, list), "Should return list of AOI files"

# Integration tests
class TestEnergyScapeIntegration:
    """Integration tests for complete workflows."""
    
    def test_synthetic_workflow(self):
        """Test complete workflow with synthetic data."""
        # This would test the full pipeline with synthetic DEM and AOI data
        # Skipped for now due to complexity
        pass
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        from config.energyscape_config import EnergyScapeManager
        
        manager = EnergyScapeManager()
        
        if manager.energyscape_config:
            is_valid = manager.validate_configuration()
            assert isinstance(is_valid, bool), "Validation should return boolean"

# Utility functions for testing
def create_synthetic_dem(rows: int = 100, cols: int = 100) -> np.ndarray:
    """Create synthetic DEM for testing."""
    x = np.linspace(0, 10, cols)
    y = np.linspace(0, 10, rows)
    X, Y = np.meshgrid(x, y)
    
    # Create varied terrain
    dem = 100 + 20 * np.sin(X) + 15 * np.cos(Y) + 5 * np.random.random((rows, cols))
    return dem

def create_synthetic_aoi():
    """Create synthetic AOI for testing."""
    from shapely.geometry import Polygon
    import geopandas as gpd
    
    # Simple rectangular AOI
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs='EPSG:4326')
    gdf['area_km2'] = 100.0
    gdf['study_site'] = 'Test Site'
    
    return gdf

# Run tests if executed directly
if __name__ == "__main__":
    print("🧪 Running EnergyScape Tests")
    print("=" * 50)
    
    # Run basic tests
    test_classes = [
        TestPontzerEquations,
        TestSlopeCalculation,
        TestEnergyScapeConfig,
        TestRIntegration,
        TestEnergyScapeCore
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"   ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"   ❌ {method_name}: {e}")
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")