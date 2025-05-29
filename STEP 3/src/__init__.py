"""EnergyScape Implementation - Source Package"""

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
