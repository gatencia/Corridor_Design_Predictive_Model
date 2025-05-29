#!/usr/bin/env python3
"""
Script to create all necessary __init__.py files for Step 3 EnergyScape implementation.
"""

from pathlib import Path

def create_init_files():
    """Create all __init__.py files with correct content."""
    
    # Define the Step 3 root directory
    step3_root = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/STEP 3")
    
    init_files = {
        # STEP 3/src/__init__.py
        step3_root / "src" / "__init__.py": '''"""EnergyScape Implementation - Source Package"""

__version__ = "0.1.0"
__author__ = "Guillaume Atencia"

# Make main classes available at package level
try:
    from .energyscape.core import EnergyScapeProcessor
    from .energyscape.pontzer_equations import PontzerEquations
    from .energyscape.slope_calculation import SlopeCalculator
    from .energyscape.r_integration import EnerscapeRBridge, PythonEnergyCalculator
    from .dem_processing.dem_loader import DEMLoader
    from .dem_processing.dem_preprocessing import DEMPreprocessor
    
    __all__ = [
        'EnergyScapeProcessor',
        'PontzerEquations', 
        'SlopeCalculator',
        'EnerscapeRBridge',
        'PythonEnergyCalculator',
        'DEMLoader',
        'DEMPreprocessor'
    ]
    
except ImportError as e:
    # Graceful handling of import errors during development
    print(f"Warning: Could not import some EnergyScape components: {e}")
    __all__ = []
''',

        # STEP 3/src/energyscape/__init__.py
        step3_root / "src" / "energyscape" / "__init__.py": '''"""EnergyScape core energy landscape calculations"""

try:
    from .core import EnergyScapeProcessor
    from .pontzer_equations import PontzerEquations
    from .slope_calculation import SlopeCalculator
    from .r_integration import EnerscapeRBridge, PythonEnergyCalculator, get_preferred_energy_calculator
    
    __all__ = [
        'EnergyScapeProcessor',
        'PontzerEquations',
        'SlopeCalculator', 
        'EnerscapeRBridge',
        'PythonEnergyCalculator',
        'get_preferred_energy_calculator'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import EnergyScape core components: {e}")
    __all__ = []
''',

        # STEP 3/src/dem_processing/__init__.py
        step3_root / "src" / "dem_processing" / "__init__.py": '''"""DEM processing utilities"""

try:
    from .dem_loader import DEMLoader, DEMInfo
    from .dem_preprocessing import DEMPreprocessor
    
    __all__ = [
        'DEMLoader',
        'DEMInfo', 
        'DEMPreprocessor'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import DEM processing components: {e}")
    __all__ = []
''',

        # STEP 3/src/validation/__init__.py
        step3_root / "src" / "validation" / "__init__.py": '''"""Validation utilities"""

# Placeholder for validation modules
__all__ = []
''',

        # STEP 3/src/utils/__init__.py
        step3_root / "src" / "utils" / "__init__.py": '''"""Utility functions"""

# Placeholder for utility modules  
__all__ = []
''',

        # STEP 3/tests/__init__.py
        step3_root / "tests" / "__init__.py": '''"""Test package for EnergyScape"""
''',

        # STEP 3/tests/fixtures/__init__.py
        step3_root / "tests" / "fixtures" / "__init__.py": '''"""Test fixtures for EnergyScape testing"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

def create_synthetic_dem(rows: int = 100, cols: int = 100, 
                        base_elevation: float = 100.0) -> np.ndarray:
    """Create synthetic DEM for testing."""
    x = np.linspace(0, 10, cols)
    y = np.linspace(0, 10, rows)
    X, Y = np.meshgrid(x, y)
    
    # Create varied terrain with different features
    dem = (base_elevation + 
           20 * np.sin(X) +           # E-W ridges
           15 * np.cos(Y) +           # N-S valleys  
           10 * np.sin(2 * X) * np.cos(2 * Y) +  # Cross pattern
           5 * np.random.random((rows, cols)))    # Noise
    
    return dem

def create_test_energy_surface(rows: int = 50, cols: int = 50) -> np.ndarray:
    """Create synthetic energy surface for testing."""
    # Base on synthetic slope pattern
    dem = create_synthetic_dem(rows, cols)
    
    # Simple slope calculation
    dy, dx = np.gradient(dem, 30.0)  # 30m resolution
    slopes = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Simple energy cost (mass=4000kg, basic Pontzer equation)
    base_cost = 1.0 * (4000 ** 0.75)  # α × M^β
    slope_multiplier = 1 + 1.5 * np.abs(np.sin(np.radians(slopes)))  # 1 + γ × |sin(θ)|
    energy_surface = base_cost * slope_multiplier
    
    return energy_surface

__all__ = [
    'create_synthetic_dem',
    'create_test_energy_surface'
]
''',
    }
    
    print("🔧 Creating __init__.py files for Step 3 EnergyScape...")
    print("=" * 60)
    
    for file_path, content in init_files.items():
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Created: {file_path.relative_to(step3_root)}")
    
    print(f"\n🎉 Successfully created {len(init_files)} __init__.py files!")
    print("\nNext steps:")
    print("1. The import errors in your code should now be resolved")
    print("2. Run the corrected EnergyScape implementation")
    print("3. It will automatically find your Step 2 AOI outputs")

if __name__ == "__main__":
    create_init_files()