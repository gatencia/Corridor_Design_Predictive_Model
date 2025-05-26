# Project configuration settings
#!/usr/bin/env python3
"""
Project Configuration for Elephant Corridor Analysis
Centralized configuration management with environment variable support.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ProjectPaths:
    """Project directory structure configuration."""
    
    # Base directories
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    src_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    
    # Data subdirectories
    data_raw_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)
    data_outputs_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.project_root / "data"
        self.src_dir = self.project_root / "src"
        self.outputs_dir = self.project_root / "outputs"
        self.reports_dir = self.project_root / "reports"
        self.logs_dir = self.project_root / "logs"
        
        # Data subdirectories
        self.data_raw_dir = self.data_dir / "raw"
        self.data_processed_dir = self.data_dir / "processed"
        self.data_outputs_dir = self.data_dir / "outputs"
    
    def create_directories(self) -> None:
        """Create all project directories if they don't exist."""
        directories = [
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_outputs_dir,
            self.outputs_dir,
            self.reports_dir / "figures",
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass
class ElephantParameters:
    """Elephant-specific parameters from literature."""
    
    # Body mass parameters (Berti et al. 2025)
    mass_female_kg: float = 2744.0
    mass_male_kg: float = 6029.0
    mass_default_kg: float = 4000.0  # Average adult
    
    # Movement parameters
    max_speed_kmh: float = 15.0
    typical_speed_kmh: float = 3.5
    daily_range_km: float = 25.0
    
    # Physiological parameters
    stride_length_m: float = 2.5
    energy_efficiency: float = 0.85
    
    def get_mass_by_sex(self, sex: str) -> float:
        """Get mass by elephant sex."""
        sex_lower = sex.lower()
        if sex_lower in ['f', 'female']:
            return self.mass_female_kg
        elif sex_lower in ['m', 'male']:
            return self.mass_male_kg
        else:
            return self.mass_default_kg

@dataclass
class EnergyLandscapeConfig:
    """Configuration for energy landscape calculations."""
    
    # Pontzer equation parameters
    alpha: float = 1.0  # Base metabolic coefficient
    beta: float = 0.75  # Mass scaling exponent
    gamma: float = 1.5  # Slope penalty coefficient
    
    # Terrain parameters
    max_traversable_slope_deg: float = 30.0
    slope_threshold_steep_deg: float = 15.0
    
    # NDVI integration
    ndvi_gain_factor: float = 1000.0  # kcal conversion
    ndvi_weight: float = 0.3
    
    # Cost surface processing
    cost_multiplier: float = 1.0
    cost_smoothing_sigma: float = 1.0
    
    def calculate_energy_cost(self, slope_deg: float, mass_kg: float) -> float:
        """Calculate energy cost using Pontzer's equation."""
        import numpy as np
        
        slope_rad = np.radians(slope_deg)
        base_cost = self.alpha * (mass_kg ** self.beta)
        slope_penalty = 1 + self.gamma * abs(np.sin(slope_rad))
        
        return base_cost * slope_penalty * self.cost_multiplier

@dataclass
class GISConfig:
    """GIS processing configuration."""
    
    # Coordinate reference systems
    default_geographic_crs: str = "EPSG:4326"  # WGS84
    default_projected_crs: str = "EPSG:32633"  # UTM Zone 33N (Central/East Africa)
    
    # Raster processing
    default_resolution_m: float = 30.0
    buffer_distance_m: float = 5000.0
    nodata_value: float = -9999.0
    
    # DEM processing
    dem_fill_voids: bool = True
    dem_smooth_iterations: int = 1
    
    # Vector processing
    simplify_tolerance_m: float = 10.0
    
    @property
    def utm_zones_africa(self) -> Dict[str, str]:
        """Common UTM zones for African elephant habitats."""
        return {
            'west': 'EPSG:32628',    # UTM 28N - West Africa
            'central': 'EPSG:32633', # UTM 33N - Central Africa  
            'east': 'EPSG:32636',    # UTM 36N - East Africa
            'south': 'EPSG:32735'    # UTM 35S - Southern Africa
        }

@dataclass
class CircuitscapeConfig:
    """Circuitscape analysis configuration."""
    
    # Analysis parameters
    scenario: str = "pairwise"  # pairwise, advanced, or all-to-one
    solver: str = "cg+amg"      # Solver algorithm
    
    # Memory management
    max_memory_gb: int = 8
    use_64bit_indexing: bool = True
    
    # Processing options
    parallelize: bool = True
    max_parallel: int = 4
    
    # Output options
    write_cur_maps: bool = True
    write_volt_maps: bool = True
    compress_grids: bool = True
    
    # Resistance processing
    resistance_is_conductance: bool = False
    use_included_pairs: bool = False
    
    def generate_ini_config(self, 
                          habitat_file: str,
                          point_file: str,
                          output_file: str) -> Dict[str, Any]:
        """Generate Circuitscape INI configuration."""
        return {
            'circuitscape_options': {
                'data_type': 'raster',
                'scenario': self.scenario,
                'solver': self.solver,
                'max_memory': self.max_memory_gb * 1000,  # Convert to MB
            },
            'habitat_file': habitat_file,
            'point_file': point_file,
            'output_file': output_file,
            'options': {
                'parallelize': self.parallelize,
                'max_parallel': self.max_parallel,
                'write_cur_maps': self.write_cur_maps,
                'write_volt_maps': self.write_volt_maps,
                'compress_grids': self.compress_grids
            }
        }

@dataclass
class VisualizationConfig:
    """Visualization and plotting configuration."""
    
    # Figure parameters
    figure_dpi: int = 300
    figure_width: float = 12.0
    figure_height: float = 8.0
    
    # Map parameters
    map_center_lat: float = 0.0   # Update for your study area
    map_center_lon: float = 25.0  # Update for your study area
    map_zoom_level: int = 8
    
    # Color schemes
    corridor_colormap: str = "viridis"
    elevation_colormap: str = "terrain"
    energy_colormap: str = "plasma"
    current_colormap: str = "YlOrRd"
    
    # Interactive map settings
    tile_layer: str = "OpenStreetMap"
    show_scale: bool = True
    show_fullscreen: bool = True
    
    # Animation parameters
    animation_fps: int = 10
    animation_duration_sec: float = 30.0
    
    @property
    def folium_tile_options(self) -> Dict[str, str]:
        """Available tile layers for Folium maps."""
        return {
            'OpenStreetMap': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            'Terrain': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
            'CartoDB': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
        }

@dataclass 
class ProcessingConfig:
    """Data processing and analysis configuration."""
    
    # Parallel processing
    max_workers: int = 4
    chunk_size: int = 1000
    use_multiprocessing: bool = True
    
    # Memory management
    memory_limit_gb: int = 16
    cache_intermediate_results: bool = True
    
    # Quality control
    gps_speed_filter_kmh: float = 100.0  # Remove unrealistic speeds
    gps_min_interval_minutes: int = 5    # Minimum time between fixes
    gps_max_gap_hours: int = 24         # Maximum gap to interpolate
    
    # Validation parameters
    cross_validation_folds: int = 5
    test_split_ratio: float = 0.2
    random_seed: int = 42

@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    
    # Log file settings
    log_file: str = "logs/elephant_corridors.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Logger names
    main_logger: str = "elephant_corridors"
    data_logger: str = "elephant_corridors.data"
    analysis_logger: str = "elephant_corridors.analysis"
    viz_logger: str = "elephant_corridors.visualization"

class ProjectConfig:
    """Main project configuration class."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize project configuration."""
        
        # Initialize configuration components
        self.paths = ProjectPaths()
        self.elephant = ElephantParameters()
        self.energy = EnergyLandscapeConfig()
        self.gis = GISConfig()
        self.circuitscape = CircuitscapeConfig()
        self.visualization = VisualizationConfig()
        self.processing = ProcessingConfig()
        self.logging = LoggingConfig()
        
        # Load custom configuration if provided
        if config_file and os.path.exists(config_file):
            self.load_config_file(config_file)
        
        # Override with environment variables
        self.load_env_variables()
        
        # Create necessary directories
        self.paths.create_directories()
    
    def load_config_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update configuration objects with loaded data
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section_obj = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def load_env_variables(self) -> None:
        """Load configuration from environment variables."""

        def get_clean_env_var(key: str) -> Optional[str]:
            val = os.getenv(key)
            if val is not None:
                return val.split('#')[0].strip()
            return None

        # Elephant parameters
        female_mass_str = get_clean_env_var('ELEPHANT_MASS_FEMALE')
        if female_mass_str:
            self.elephant.mass_female_kg = float(female_mass_str)
        
        male_mass_str = get_clean_env_var('ELEPHANT_MASS_MALE')
        if male_mass_str:
            self.elephant.mass_male_kg = float(male_mass_str)
        
        # GIS parameters
        default_crs_str = get_clean_env_var('DEFAULT_CRS')
        if default_crs_str: # Assuming DEFAULT_CRS does not need float conversion
            self.gis.default_projected_crs = default_crs_str
        
        buffer_distance_str = get_clean_env_var('DEFAULT_BUFFER_DISTANCE')
        if buffer_distance_str:
            self.gis.buffer_distance_m = float(buffer_distance_str)
        
        dem_resolution_str = get_clean_env_var('DEFAULT_DEM_RESOLUTION')
        if dem_resolution_str:
            self.gis.default_resolution_m = float(dem_resolution_str)
        
        # Map center
        map_center_lat_str = get_clean_env_var('MAP_CENTER_LAT')
        if map_center_lat_str:
            self.visualization.map_center_lat = float(map_center_lat_str)
        
        map_center_lon_str = get_clean_env_var('MAP_CENTER_LON')
        if map_center_lon_str:
            self.visualization.map_center_lon = float(map_center_lon_str)
        
        # Processing parameters
        max_workers_str = get_clean_env_var('MAX_WORKERS')
        if max_workers_str:
            self.processing.max_workers = int(max_workers_str)
            
        memory_limit_gb_str = get_clean_env_var('MEMORY_LIMIT_GB')
        if memory_limit_gb_str:
            self.processing.memory_limit_gb = int(memory_limit_gb_str)
        
        # Logging
        log_level_str = get_clean_env_var('LOG_LEVEL')
        if log_level_str: # Assuming LOG_LEVEL does not need float conversion
            self.logging.level = log_level_str
    
    def save_config(self, output_file: str) -> None:
        """Save current configuration to YAML file."""
        config_dict = {
            'elephant': self.elephant.__dict__,
            'energy': self.energy.__dict__, 
            'gis': self.gis.__dict__,
            'circuitscape': self.circuitscape.__dict__,
            'visualization': self.visualization.__dict__,
            'processing': self.processing.__dict__,
            'logging': self.logging.__dict__
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_study_area_crs(self, center_lon: float) -> str:
        """Determine appropriate UTM CRS based on longitude."""
        if center_lon < -6:
            return self.gis.utm_zones_africa['west']
        elif center_lon < 15:
            return self.gis.utm_zones_africa['central']
        elif center_lon < 30:
            return self.gis.utm_zones_africa['east']
        else:
            return self.gis.utm_zones_africa['south']
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Check required directories exist
            assert self.paths.project_root.exists(), "Project root does not exist"
            
            # Check parameter ranges
            assert 0 < self.elephant.mass_female_kg < 10000, "Invalid female mass"
            assert 0 < self.elephant.mass_male_kg < 10000, "Invalid male mass"
            assert 0 < self.gis.default_resolution_m < 1000, "Invalid resolution"
            assert -90 <= self.visualization.map_center_lat <= 90, "Invalid latitude"
            assert -180 <= self.visualization.map_center_lon <= 180, "Invalid longitude"
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("🐘 Elephant Corridor Analysis - Configuration Summary")
        print("=" * 60)
        print(f"Project Root: {self.paths.project_root}")
        print(f"Default CRS: {self.gis.default_projected_crs}")
        print(f"DEM Resolution: {self.gis.default_resolution_m}m")
        print(f"Buffer Distance: {self.gis.buffer_distance_m}m")
        print(f"Female Mass: {self.elephant.mass_female_kg}kg")
        print(f"Male Mass: {self.elephant.mass_male_kg}kg")
        print(f"Max Workers: {self.processing.max_workers}")
        print(f"Memory Limit: {self.processing.memory_limit_gb}GB")
        print(f"Log Level: {self.logging.level}")
        print("=" * 60)

# Global configuration instance
config = ProjectConfig()

# Convenience functions for accessing configuration
def get_config() -> ProjectConfig:
    """Get the global configuration instance."""
    return config

def update_config(config_file: str) -> None:
    """Update global configuration from file."""
    global config
    config = ProjectConfig(config_file)

def get_paths() -> ProjectPaths:
    """Get project paths configuration."""
    return config.paths

def get_elephant_params() -> ElephantParameters:
    """Get elephant parameters."""
    return config.elephant

def get_gis_config() -> GISConfig:
    """Get GIS configuration."""
    return config.gis

def get_energy_config() -> EnergyLandscapeConfig:
    """Get energy landscape configuration."""
    return config.energy

# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = ProjectConfig()
    
    print("Testing configuration...")
    test_config.print_summary()
    
    # Validate configuration
    if test_config.validate_config():
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
    
    # Test energy cost calculation
    print(f"\nEnergy cost examples:")
    print(f"Female elephant, flat terrain: {test_config.energy.calculate_energy_cost(0, test_config.elephant.mass_female_kg):.2f}")
    print(f"Female elephant, 15° slope: {test_config.energy.calculate_energy_cost(15, test_config.elephant.mass_female_kg):.2f}")
    print(f"Male elephant, 15° slope: {test_config.energy.calculate_energy_cost(15, test_config.elephant.mass_male_kg):.2f}")
    
    # Test CRS selection
    print(f"\nCRS selection examples:")
    print(f"Longitude -10°: {test_config.get_study_area_crs(-10)}")
    print(f"Longitude 10°: {test_config.get_study_area_crs(10)}")
    print(f"Longitude 25°: {test_config.get_study_area_crs(25)}")
    print(f"Longitude 35°: {test_config.get_study_area_crs(35)}")
    
    print("\n🎉 Configuration module working correctly!")