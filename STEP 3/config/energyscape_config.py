#!/usr/bin/env python3
"""
EnergyScape-specific configuration.
Extends the main project configuration with energy landscape parameters.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import yaml

# Add STEP 1 to path for ProjectConfig
step1_dir = Path(__file__).parent.parent.parent / "STEP 1"
sys.path.insert(0, str(step1_dir))

try:
    from config.project_config import ProjectConfig, get_config
except ImportError:
    # Fallback if Step 1 config not available
    class ProjectConfig:
        def __init__(self):
            pass
    
    def get_config():
        return ProjectConfig()

@dataclass
class EnergyScapeConfig:
    """Configuration for EnergyScape energy landscape calculations."""
    
    # Pontzer equation parameters (from Berti et al. 2025)
    pontzer_alpha: float = 1.0      # Base metabolic coefficient
    pontzer_beta: float = 0.75      # Mass scaling exponent (allometric)
    pontzer_gamma: float = 1.5      # Slope penalty coefficient
    
    # Elephant body masses (kg) - from Berti et al. 2025
    elephant_mass_female: float = 2744.0
    elephant_mass_male: float = 6029.0
    elephant_mass_average: float = 4000.0
    
    # DEM processing parameters
    dem_resolution_m: float = 30.0           # Target DEM resolution (NASADEM)
    dem_nodata_value: float = -9999.0        # NoData value for DEM
    dem_fill_voids: bool = True              # Fill small voids in DEM
    dem_smooth_iterations: int = 1           # Smoothing iterations
    
    # Slope calculation parameters
    max_slope_degrees: float = 45.0          # Maximum processable slope
    slope_calculation_method: str = "numpy"   # Method: 'numpy', 'gdal', or 'richdem'
    
    # Energy surface parameters
    energy_cost_multiplier: float = 1.0      # Global cost scaling factor
    energy_nodata_value: float = -9999.0     # NoData for energy surfaces
    
    # NDVI integration parameters
    ndvi_gain_factor: float = 1000.0          # kcal conversion factor
    ndvi_weight: float = 0.3                  # Weight for NDVI in net energy
    ndvi_nodata_value: float = -9999.0        # NoData for NDVI
    
    # R integration parameters
    r_timeout_seconds: int = 300              # R process timeout
    use_r_enerscape: bool = True              # Use R enerscape package
    r_memory_limit_mb: int = 4096             # R memory limit
    
    # Output parameters
    output_format: str = "GTiff"              # Output raster format
    output_compress: str = "LZW"              # Compression method
    output_tiled: bool = True                 # Create tiled outputs
    output_overviews: bool = True             # Generate overviews
    
    # Validation parameters
    validation_sample_points: int = 1000      # Points for validation
    validation_tolerance: float = 0.1         # Tolerance for validation
    
    # Processing parameters
    chunk_size: int = 1024                    # Processing chunk size
    max_workers: int = 4                      # Parallel processing workers
    
    @classmethod
    def from_project_config(cls, project_config: ProjectConfig) -> 'EnergyScapeConfig':
        """Create EnergyScapeConfig from main ProjectConfig."""
        config = cls()
        
        # Import elephant parameters if available
        if hasattr(project_config, 'elephant'):
            config.elephant_mass_female = project_config.elephant.mass_female_kg
            config.elephant_mass_male = project_config.elephant.mass_male_kg
            config.elephant_mass_average = project_config.elephant.mass_default_kg
        
        # Import energy parameters if available
        if hasattr(project_config, 'energy'):
            config.pontzer_alpha = project_config.energy.alpha
            config.pontzer_beta = project_config.energy.beta
            config.pontzer_gamma = project_config.energy.gamma
            config.energy_cost_multiplier = project_config.energy.cost_multiplier
        
        # Import GIS parameters if available
        if hasattr(project_config, 'gis'):
            config.dem_resolution_m = project_config.gis.default_resolution_m
            config.dem_nodata_value = project_config.gis.nodata_value
        
        # Import processing parameters if available
        if hasattr(project_config, 'processing'):
            config.max_workers = project_config.processing.max_workers
        
        return config
    
    def get_elephant_masses(self) -> Dict[str, float]:
        """Get elephant body masses by sex."""
        return {
            'female': self.elephant_mass_female,
            'male': self.elephant_mass_male,
            'average': self.elephant_mass_average
        }
    
    def get_pontzer_parameters(self) -> Dict[str, float]:
        """Get Pontzer equation parameters."""
        return {
            'alpha': self.pontzer_alpha,
            'beta': self.pontzer_beta,
            'gamma': self.pontzer_gamma
        }

class EnergyScapeManager:
    """Manages EnergyScape configuration and integration with main project config."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize EnergyScape configuration manager.
        
        Parameters:
        -----------
        config_file : str, optional
            Path to YAML configuration file for EnergyScape overrides
        """
        # Load main project configuration
        self.project_config = get_config()
        
        # Create EnergyScape configuration
        self.energyscape_config = EnergyScapeConfig.from_project_config(self.project_config)
        
        # Override with YAML file if provided
        if config_file and Path(config_file).exists():
            self.load_yaml_config(config_file)
    
    def load_yaml_config(self, config_file: str) -> None:
        """Load EnergyScape configuration from YAML file."""
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Update EnergyScape config with YAML values
        if 'energyscape' in yaml_config:
            energyscape_section = yaml_config['energyscape']
            for key, value in energyscape_section.items():
                if hasattr(self.energyscape_config, key):
                    setattr(self.energyscape_config, key, value)
    
    def save_yaml_config(self, output_file: str) -> None:
        """Save current EnergyScape configuration to YAML file."""
        config_dict = {
            'energyscape': {
                field.name: getattr(self.energyscape_config, field.name)
                for field in self.energyscape_config.__dataclass_fields__.values()
            }
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_step2_outputs_dir(self) -> Path:
        """Get Step 2 outputs directory."""
        step2_dir = Path(__file__).parent.parent.parent / "STEP 2" / "data" / "outputs"
        return step2_dir
    
    def get_step3_data_dirs(self) -> Dict[str, Path]:
        """Get Step 3 data directories."""
        step3_root_dir = Path(__file__).parent.parent  # This is STEP 3/
        project_root_dir = step3_root_dir.parent      # This is Elephant_Corridor_Research/
        
        step2_5_output_dir = project_root_dir / "STEP 2.5" / "outputs" / "aoi_specific_dems"
        
        return {
            'raw_dem': step2_5_output_dir,  # Changed to point to STEP 2.5 outputs
            'raw_ndvi': step3_root_dir / "data" / "raw" / "ndvi",
            'processed_dem': step3_root_dir / "data" / "processed" / "dem_clipped",
            'energy_surfaces': step3_root_dir / "data" / "processed" / "energy_surfaces",
            'outputs': step3_root_dir / "data" / "outputs",
            'net_energy': step3_root_dir / "data" / "outputs" / "net_energy_surfaces"
        }
    
    def validate_configuration(self) -> bool:
        """Validate EnergyScape configuration parameters."""
        try:
            config = self.energyscape_config
            
            # Validate elephant masses
            assert config.elephant_mass_female > 0, "Female mass must be positive"
            assert config.elephant_mass_male > 0, "Male mass must be positive"
            assert config.elephant_mass_male > config.elephant_mass_female, "Male mass should be larger than female"
            
            # Validate Pontzer parameters
            assert config.pontzer_alpha > 0, "Alpha must be positive"
            assert config.pontzer_beta > 0, "Beta must be positive"
            assert config.pontzer_gamma > 0, "Gamma must be positive"
            
            # Validate DEM parameters
            assert config.dem_resolution_m > 0, "DEM resolution must be positive"
            assert 0 < config.max_slope_degrees <= 90, "Max slope must be between 0-90 degrees"
            
            # Validate processing parameters
            assert config.max_workers > 0, "Max workers must be positive"
            assert config.chunk_size > 0, "Chunk size must be positive"
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def print_summary(self) -> None:
        """Print EnergyScape configuration summary."""
        config = self.energyscape_config
        
        print("⚡ EnergyScape Configuration Summary")
        print("=" * 60)
        print(f"Elephant Masses:")
        print(f"  Female: {config.elephant_mass_female} kg")
        print(f"  Male: {config.elephant_mass_male} kg")
        print(f"  Average: {config.elephant_mass_average} kg")
        print(f"\nPontzer Parameters:")
        print(f"  Alpha (base): {config.pontzer_alpha}")
        print(f"  Beta (mass scaling): {config.pontzer_beta}")
        print(f"  Gamma (slope penalty): {config.pontzer_gamma}")
        print(f"\nDEM Processing:")
        print(f"  Resolution: {config.dem_resolution_m}m")
        print(f"  Max slope: {config.max_slope_degrees}°")
        print(f"  Fill voids: {config.dem_fill_voids}")
        print(f"\nProcessing:")
        print(f"  Max workers: {config.max_workers}")
        print(f"  Chunk size: {config.chunk_size}")
        print(f"  Use R enerscape: {config.use_r_enerscape}")
        print("=" * 60)

# Global configuration instance
_energyscape_manager = None

def get_energyscape_config(config_file: Optional[str] = None) -> EnergyScapeConfig:
    """Get global EnergyScape configuration instance."""
    global _energyscape_manager
    if _energyscape_manager is None:
        _energyscape_manager = EnergyScapeManager(config_file)
    return _energyscape_manager.energyscape_config

def get_energyscape_manager(config_file: Optional[str] = None) -> EnergyScapeManager:
    """Get global EnergyScape manager instance."""
    global _energyscape_manager
    if _energyscape_manager is None:
        _energyscape_manager = EnergyScapeManager(config_file)
    return _energyscape_manager

# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    manager = EnergyScapeManager()
    
    print("Testing EnergyScape configuration...")
    manager.print_summary()
    
    # Validate configuration
    if manager.validate_configuration():
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
    
    # Test elephant masses
    masses = manager.energyscape_config.get_elephant_masses()
    print(f"\nElephant masses: {masses}")
    
    # Test Pontzer parameters
    pontzer_params = manager.energyscape_config.get_pontzer_parameters()
    print(f"Pontzer parameters: {pontzer_params}")
    
    # Test directory paths
    step2_dir = manager.get_step2_outputs_dir()
    step3_dirs = manager.get_step3_data_dirs()
    print(f"\nStep 2 outputs directory: {step2_dir}")
    print(f"Step 3 data directories: {list(step3_dirs.keys())}")
    
    print("\n🎉 EnergyScape configuration module working correctly!")