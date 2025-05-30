#!/usr/bin/env python3
"""
Comprehensive Corridor Prediction Analysis
Comparing Energyscape vs Traditional Raster-based Corridor Models

This script implements a complete workflow to:
1. Load GPS data and AOIs from Step 2
2. Generate synthetic raster layers
3. Implement both Energyscape and traditional corridor models
4. Compare accuracy and validate against GPS tracking data
5. Generate clear visualizations and deliverables

Run from: Temp Fake Data folder
Requirements: pandas, geopandas, numpy, rasterio, matplotlib, scipy, skimage
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio import features, transform
from rasterio.enums import Resampling
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from skimage import morphology
from skimage.segmentation import watershed

# Set up paths and create directory structure
TEMP_DATA_DIR = Path("Temp_Fake_Data")
TEMP_DATA_DIR.mkdir(exist_ok=True)

# Create subdirectories
(TEMP_DATA_DIR / "rasters").mkdir(exist_ok=True)
(TEMP_DATA_DIR / "corridors").mkdir(exist_ok=True)
(TEMP_DATA_DIR / "visualizations").mkdir(exist_ok=True)
(TEMP_DATA_DIR / "results").mkdir(exist_ok=True)

class CorridorPredictionAnalysis:
    """
    Comprehensive corridor prediction analysis comparing Energyscape and traditional methods.
    """
    
    def __init__(self, step2_data_path=None):
        """Initialize the analysis with Step 2 data path."""
        self.step2_data_path = Path(step2_data_path) if step2_data_path else Path("../STEP 2")
        self.temp_dir = TEMP_DATA_DIR
        self.raster_dir = self.temp_dir / "rasters"
        self.results_dir = self.temp_dir / "results"
        
        # Analysis parameters
        self.raster_resolution = 100  # 100m resolution
        self.elephant_mass_kg = 4000  # Forest elephant mass
        self.buffer_distance_m = 2000  # Buffer around corridors for validation
        
        # Results containers
        self.gps_data = None
        self.aoi_bounds = None
        self.raster_profile = None
        self.raster_layers = {}
        self.corridor_predictions = {}
        self.validation_results = {}
        self.aoi_area_km2 = None  # Initialize AOI area
        
        print(f"🐘 Corridor Prediction Analysis Initialized")
        print(f"📁 Working directory: {self.temp_dir}")
        print(f"📊 Raster resolution: {self.raster_resolution}m")
        
    def _get_study_duration_info(self):
        """Helper function to safely calculate study duration days and text."""
        if not (hasattr(self, 'gps_data') and self.gps_data is not None and \
                'timestamp' in self.gps_data.columns):
            return None, "N/A (timestamp data unavailable)"

        # Ensure gps_data and timestamp column are not effectively empty
        if self.gps_data.empty or self.gps_data['timestamp'].isnull().all():
             return None, "N/A (timestamp data empty or all null)"

        timestamps_data = self.gps_data['timestamp']
        
        # Ensure it's pandas Series for methods like .empty and .dropna()
        if not isinstance(timestamps_data, pd.Series):
            try:
                timestamps_data = pd.Series(timestamps_data)
            except Exception:
                return None, "N/A (timestamp data not series-like)"

        if not pd.api.types.is_datetime64_any_dtype(timestamps_data):
            timestamps_data = pd.to_datetime(timestamps_data, errors='coerce')
        
        if not pd.api.types.is_datetime64_any_dtype(timestamps_data): # Still not datetime
            return None, "N/A (timestamp conversion failed)"

        valid_timestamps = timestamps_data.dropna()
        if valid_timestamps.empty:
            return None, "N/A (no valid timestamps after conversion)"
            
        min_ts = valid_timestamps.min()
        max_ts = valid_timestamps.max()
        
        if pd.isna(min_ts) or pd.isna(max_ts): 
             return None, "N/A (invalid min/max date after conversion)"

        try:
            if not isinstance(min_ts, pd.Timestamp) or not isinstance(max_ts, pd.Timestamp):
                 return None, "N/A (min/max not proper Timestamps)"

            duration_delta = max_ts - min_ts
            duration_days = duration_delta.days
            
            if pd.isna(duration_days): 
                return None, "N/A (could not calculate days from duration)"
            return duration_days, f"{duration_days} days"
        except (TypeError, AttributeError): 
            return None, "N/A (error during duration calculation)"

    def load_step2_data(self):
        """Load GPS data and AOIs from Step 2 processing."""
        print(f"\n📂 Loading Step 2 Data...")
        
        # Find GPS data files from Step 2 outputs
        gps_files = []
        aoi_files = []
        
        # Look for processed data in Step 2 outputs
        step2_outputs = self.step2_data_path / "data" / "outputs"
        if step2_outputs.exists():
            # Load individual AOI data
            individual_aois = step2_outputs / "individual_aois"
            if individual_aois.exists():
                for aoi_folder in individual_aois.iterdir():
                    if aoi_folder.is_dir():
                        # Find CSV and GeoJSON files
                        csv_files = list(aoi_folder.glob("gps_points_*.csv"))
                        geojson_files = list(aoi_folder.glob("aoi_*.geojson"))
                        
                        if csv_files and geojson_files:
                            gps_files.extend(csv_files)
                            aoi_files.extend(geojson_files)
        
        if not gps_files or not aoi_files:
            print("⚠️  No Step 2 processed data found, using sample data...")
            self._create_sample_data()
            return
        
        # Load the first available dataset for demonstration
        print(f"📄 Found {len(gps_files)} GPS files and {len(aoi_files)} AOI files")
        
        # Load GPS data
        gps_df = pd.read_csv(gps_files[0])
        # Ensure timestamp is converted early
        if 'timestamp' in gps_df.columns:
            gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp'], errors='coerce')

        self.gps_data = gpd.GeoDataFrame(
            gps_df,
            geometry=gpd.points_from_xy(gps_df['longitude'], gps_df['latitude']),
            crs='EPSG:4326'
        )
        
        # Load AOI
        aoi_gdf = gpd.read_file(aoi_files[0])
        self.aoi_area_km2 = aoi_gdf['area_km2'].iloc[0] # Store AOI area
        
        # Get AOI bounds for raster generation
        aoi_utm = aoi_gdf.to_crs('EPSG:32633')  # UTM 33N for Cameroon
        self.aoi_bounds = aoi_utm.total_bounds
        
        print(f"✅ Loaded {len(self.gps_data)} GPS points")
        
        _, study_duration_text = self._get_study_duration_info() # Use helper
        print(f"📊 Study period: {study_duration_text}")
            
        print(f"🗺️  AOI area: {self.aoi_area_km2:.1f} km²") # Use stored value
        
        # Set up raster profile
        self._setup_raster_profile()
    
    def _create_sample_data(self):
        """Create sample data if Step 2 data is not available."""
        
        # Create sample GPS track
        n_points = 200
        np.random.seed(42)
        
        # Simulate realistic elephant movement
        center_lat, center_lon = 8.5, 13.5
        
        # Random walk with some directional bias
        lat_steps = np.random.normal(0, 0.01, n_points) + np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.005
        lon_steps = np.random.normal(0, 0.01, n_points) + np.cos(np.linspace(0, 4*np.pi, n_points)) * 0.005
        
        latitudes = np.cumsum(lat_steps) + center_lat
        longitudes = np.cumsum(lon_steps) + center_lon
        
        # Create GPS DataFrame
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='3H')
        
        gps_df = pd.DataFrame({
            'timestamp': timestamps, # Already datetime objects
            'latitude': latitudes,
            'longitude': longitudes,
            'individual-local-identifier': [1] * n_points
        })
        
        self.gps_data = gpd.GeoDataFrame(
            gps_df,
            geometry=gpd.points_from_xy(longitudes, latitudes),
            crs='EPSG:4326'
        )
        
        # Create AOI bounds (10km buffer around GPS points)
        gps_utm = self.gps_data.to_crs('EPSG:32633')
        bounds = gps_utm.total_bounds
        buffer = 10000  # 10km buffer
        self.aoi_bounds = [
            bounds[0] - buffer, bounds[1] - buffer,
            bounds[2] + buffer, bounds[3] + buffer
        ]
        # Calculate and store AOI area for sample data
        self.aoi_area_km2 = (self.aoi_bounds[2] - self.aoi_bounds[0]) * \
                            (self.aoi_bounds[3] - self.aoi_bounds[1]) / 1_000_000
        
        print(f"✅ Created sample dataset with {len(self.gps_data)} GPS points")
        self._setup_raster_profile()
    
    def _setup_raster_profile(self):
        """Setup raster profile for all synthetic layers."""
        
        # Calculate raster dimensions
        width = int((self.aoi_bounds[2] - self.aoi_bounds[0]) / self.raster_resolution)
        height = int((self.aoi_bounds[3] - self.aoi_bounds[1]) / self.raster_resolution)
        
        # Create affine transformation
        raster_transform = transform.from_bounds(
            self.aoi_bounds[0], self.aoi_bounds[1],
            self.aoi_bounds[2], self.aoi_bounds[3],
            width, height
        )
        
        self.raster_profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999.0,
            'width': width,
            'height': height,
            'count': 1,
            'crs': 'EPSG:32633',
            'transform': raster_transform
        }
        
        print(f"📐 Raster grid: {width} x {height} pixels")
        print(f"📏 Coverage: {(width * self.raster_resolution / 1000):.1f} x {(height * self.raster_resolution / 1000):.1f} km")
    
    def generate_synthetic_rasters(self):
        """Generate all synthetic raster layers efficiently."""
        print(f"\n🗺️  Generating Synthetic Raster Layers...")
        
        width, height = self.raster_profile['width'], self.raster_profile['height']
        
        # 1. DEM (Digital Elevation Model)
        print("   📈 Creating DEM...")
        np.random.seed(42)
        
        # Create realistic terrain with multiple peaks and valleys
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        
        # Base elevation with multiple terrain features
        dem = (
            300 + 200 * np.sin(X * 0.5) * np.cos(Y * 0.5) +  # Large scale terrain
            100 * np.sin(X * 2) * np.sin(Y * 1.5) +          # Medium scale features
            50 * np.random.random((height, width)) +          # Random variation
            150 * np.exp(-((X-5)**2 + (Y-5)**2) / 4)         # Central highland
        )
        
        # Smooth the DEM
        dem = ndimage.gaussian_filter(dem, sigma=2)
        
        self.raster_layers['dem'] = dem
        self._save_raster(dem, 'dem.tif', 'Digital Elevation Model (m)')
        
        # 2. Slope
        print("   📐 Calculating Slope...")
        
        # Calculate slope from DEM
        grad_x, grad_y = np.gradient(dem, self.raster_resolution)
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180 / np.pi  # Convert to degrees
        
        self.raster_layers['slope'] = slope
        self._save_raster(slope, 'slope.tif', 'Slope (degrees)')
        
        # 3. NDVI (Vegetation Index)
        print("   🌿 Creating NDVI...")
        
        # NDVI varies with elevation and has seasonal patterns
        base_ndvi = 0.7 - (dem - dem.min()) / (dem.max() - dem.min()) * 0.4
        seasonal_variation = 0.1 * np.sin(X * 0.3) * np.cos(Y * 0.3)
        noise = 0.05 * np.random.random((height, width))
        
        ndvi = np.clip(base_ndvi + seasonal_variation + noise, 0, 1)
        ndvi = ndimage.gaussian_filter(ndvi, sigma=1)
        
        self.raster_layers['ndvi'] = ndvi
        self._save_raster(ndvi, 'ndvi.tif', 'NDVI (0-1)')
        
        # 4. Water Availability
        print("   💧 Creating Water Availability...")
        
        # Water sources are more common in valleys and have network structure
        water = np.zeros((height, width))
        
        # Create river networks in valleys
        valleys = dem < np.percentile(dem, 30)
        water[valleys] = 0.8
        
        # Add random water sources
        n_sources = 15
        for _ in range(n_sources):
            wx, wy = np.random.randint(10, width-10), np.random.randint(10, height-10)
            water[wy-5:wy+5, wx-5:wx+5] = 1.0
        
        # Distance decay from water sources
        from scipy.ndimage import distance_transform_edt
        water_binary = water > 0.5
        distances = distance_transform_edt(~water_binary) * self.raster_resolution
        water_availability = np.exp(-distances / 2000)  # 2km decay
        
        self.raster_layers['water'] = water_availability
        self._save_raster(water_availability, 'water_availability.tif', 'Water Availability (0-1)')
        
        # 5. Human Pressure
        print("   🏘️  Creating Human Pressure...")
        
        # Human pressure decreases with distance from settlements
        human_pressure = np.zeros((height, width))
        
        # Add settlements
        n_settlements = 8
        for _ in range(n_settlements):
            sx, sy = np.random.randint(0, width), np.random.randint(0, height)
            intensity = np.random.uniform(0.5, 1.0)
            
            # Create pressure gradient around settlement
            for i in range(height):
                for j in range(width):
                    dist = np.sqrt((i - sy)**2 + (j - sx)**2) * self.raster_resolution
                    pressure = intensity * np.exp(-dist / 5000)  # 5km decay
                    human_pressure[i, j] = max(human_pressure[i, j], pressure)
        
        # Add roads (linear high pressure)
        road_y = height // 2
        human_pressure[road_y-2:road_y+3, :] = np.maximum(
            human_pressure[road_y-2:road_y+3, :], 0.6
        )
        
        self.raster_layers['human_pressure'] = human_pressure
        self._save_raster(human_pressure, 'human_pressure.tif', 'Human Pressure (0-1)')
        
        # 6. Land Cover
        print("   🌳 Creating Land Cover...")
        
        # Land cover based on elevation, slope, and human pressure
        forest = (ndvi > 0.6) & (slope < 15) & (human_pressure < 0.3)
        savanna = (ndvi > 0.4) & (ndvi <= 0.6) & (slope < 25)
        grassland = (ndvi > 0.2) & (ndvi <= 0.4)
        bare = ndvi <= 0.2
        urban = human_pressure > 0.7
        
        land_cover = np.zeros((height, width))
        land_cover[forest] = 1  # Forest
        land_cover[savanna] = 2  # Savanna
        land_cover[grassland] = 3  # Grassland
        land_cover[bare] = 4  # Bare ground
        land_cover[urban] = 5  # Urban/settlements
        
        self.raster_layers['land_cover'] = land_cover
        self._save_raster(land_cover, 'land_cover.tif', 'Land Cover Classes')
        
        # 7. Climate (Rainfall)
        print("   🌧️  Creating Climate...")
        
        # Rainfall patterns with orographic effects
        base_rainfall = 1200  # mm/year base
        orographic = (dem - dem.min()) / (dem.max() - dem.min()) * 300
        seasonal = 200 * np.cos(X * 0.2) * np.sin(Y * 0.2)
        
        rainfall = base_rainfall + orographic + seasonal
        rainfall = np.clip(rainfall, 800, 2000)  # Realistic range
        
        self.raster_layers['climate'] = rainfall
        self._save_raster(rainfall, 'rainfall.tif', 'Annual Rainfall (mm)')
        
        # 8. Resource Distribution
        print("   🥭 Creating Resource Distribution...")
        
        # Fruiting trees and mineral licks
        resources = np.zeros((height, width))
        
        # High resources in forests with good water access
        good_habitat = (land_cover == 1) & (water_availability > 0.3) & (human_pressure < 0.4)
        resources[good_habitat] = 0.8
        
        # Add specific resource patches
        n_patches = 20
        for _ in range(n_patches):
            px, py = np.random.randint(5, width-5), np.random.randint(5, height-5)
            if good_habitat[py, px]:
                resources[py-3:py+4, px-3:px+4] = 1.0
        
        resources = ndimage.gaussian_filter(resources, sigma=1)
        
        self.raster_layers['resources'] = resources
        self._save_raster(resources, 'resources.tif', 'Resource Availability (0-1)')
        
        # 9. Anthropogenic Risk
        print("   ⚠️  Creating Anthropogenic Risk...")
        
        # Risk from human activities
        risk = human_pressure.copy()
        
        # Higher risk near roads and settlements
        risk += 0.3 * (human_pressure > 0.5)
        
        # Lower risk in protected areas (simulate)
        protected_x = slice(width//4, 3*width//4)
        protected_y = slice(height//4, 3*height//4)
        risk[protected_y, protected_x] *= 0.3
        
        risk = np.clip(risk, 0, 1)
        
        self.raster_layers['risk'] = risk
        self._save_raster(risk, 'anthropogenic_risk.tif', 'Anthropogenic Risk (0-1)')
        
        print(f"✅ Generated {len(self.raster_layers)} synthetic raster layers")
    
    def _save_raster(self, data, filename, description):
        """Save raster data to file."""
        filepath = self.raster_dir / filename
        
        with rasterio.open(filepath, 'w', **self.raster_profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, description)
    
    def calculate_energyscape_model(self):
        """Calculate Energyscape-based corridor prediction."""
        print(f"\n⚡ Calculating Energyscape Model...")
        
        dem = self.raster_layers['dem']
        slope = self.raster_layers['slope']
        ndvi = self.raster_layers['ndvi']
        water = self.raster_layers['water']
        
        # Energy cost calculation based on Halsey & White (2017) and similar studies
        # Base metabolic cost (Watts) for 4000kg elephant
        basal_metabolic_rate = 3.39 * (self.elephant_mass_kg ** 0.756)  # Watts
        
        # Terrain-based energy cost
        # Cost increases exponentially with slope
        slope_cost_multiplier = 1 + np.exp(slope / 10)  # Exponential increase with slope
        
        # Elevation change cost (uphill is more expensive)
        grad_x, grad_y = np.gradient(dem, self.raster_resolution)
        elevation_gradient = np.sqrt(grad_x**2 + grad_y**2)
        elevation_cost = 1 + elevation_gradient / 50  # Normalized elevation cost
        
        # Base movement cost (energy per meter)
        base_movement_cost = basal_metabolic_rate * 0.1  # Arbitrary scaling
        
        # Total energy cost per pixel
        energy_cost_per_meter = base_movement_cost * slope_cost_multiplier * elevation_cost
        energy_cost_per_pixel = energy_cost_per_meter * self.raster_resolution
        
        # Resource availability reduces effective cost
        resource_benefit = 0.2 + 0.8 * ndvi * water  # Combined resource index
        
        # Final Energyscape resistance
        energyscape_resistance = energy_cost_per_pixel / resource_benefit
        
        # Normalize to 0-1 range
        energyscape_resistance = (energyscape_resistance - energyscape_resistance.min()) / \
                               (energyscape_resistance.max() - energyscape_resistance.min())
        
        self.raster_layers['energyscape'] = energyscape_resistance
        self._save_raster(energyscape_resistance, 'energyscape_resistance.tif', 
                         'Energyscape Resistance (0-1)')
        
        # Calculate least-cost corridors using Energyscape
        corridors = self._calculate_least_cost_paths(energyscape_resistance, 'Energyscape')
        self.corridor_predictions['energyscape'] = corridors
        
        print(f"✅ Energyscape model completed")
        print(f"   📊 Resistance range: {energyscape_resistance.min():.3f} - {energyscape_resistance.max():.3f}")
    
    def calculate_traditional_model(self):
        """Calculate traditional raster-based corridor prediction."""
        print(f"\n📊 Calculating Traditional Raster Model...")
        
        # Traditional resistance based on multiple factors
        land_cover = self.raster_layers['land_cover']
        human_pressure = self.raster_layers['human_pressure']
        slope = self.raster_layers['slope']
        water = self.raster_layers['water']
        ndvi = self.raster_layers['ndvi']
        risk = self.raster_layers['risk']
        
        # Land cover resistance values
        lc_resistance = np.zeros_like(land_cover)
        lc_resistance[land_cover == 1] = 0.1  # Forest (low resistance)
        lc_resistance[land_cover == 2] = 0.3  # Savanna
        lc_resistance[land_cover == 3] = 0.5  # Grassland
        lc_resistance[land_cover == 4] = 0.8  # Bare ground
        lc_resistance[land_cover == 5] = 1.0  # Urban (high resistance)
        
        # Slope resistance (elephants avoid steep terrain)
        slope_resistance = np.clip(slope / 30, 0, 1)  # Normalize to 0-1
        
        # Human pressure resistance
        pressure_resistance = human_pressure
        
        # Water availability (inverse - lack of water increases resistance)
        water_resistance = 1 - water
        
        # NDVI resistance (inverse - low vegetation increases resistance)
        vegetation_resistance = 1 - ndvi
        
        # Anthropogenic risk resistance
        risk_resistance = risk
        
        # Combined traditional resistance (weighted average)
        traditional_resistance = (
            0.25 * lc_resistance +
            0.20 * slope_resistance +
            0.20 * pressure_resistance +
            0.15 * water_resistance +
            0.10 * vegetation_resistance +
            0.10 * risk_resistance
        )
        
        # Normalize to 0-1 range
        traditional_resistance = (traditional_resistance - traditional_resistance.min()) / \
                               (traditional_resistance.max() - traditional_resistance.min())
        
        self.raster_layers['traditional'] = traditional_resistance
        self._save_raster(traditional_resistance, 'traditional_resistance.tif',
                         'Traditional Resistance (0-1)')
        
        # Calculate least-cost corridors using traditional method
        corridors = self._calculate_least_cost_paths(traditional_resistance, 'Traditional')
        self.corridor_predictions['traditional'] = corridors
        
        print(f"✅ Traditional model completed")
        print(f"   📊 Resistance range: {traditional_resistance.min():.3f} - {traditional_resistance.max():.3f}")
    
    def _calculate_least_cost_paths(self, resistance_surface, model_name):
        """Calculate least-cost paths from resistance surface."""
        
        # Convert GPS points to raster coordinates
        gps_utm = self.gps_data.to_crs('EPSG:32633')
        
        # Get raster coordinates for GPS points
        transform_obj = self.raster_profile['transform']
        
        gps_coords = []
        for point in gps_utm.geometry:
            col, row = ~transform_obj * (point.x, point.y)
            col, row = int(col), int(row)
            
            # Ensure coordinates are within raster bounds
            if 0 <= row < resistance_surface.shape[0] and 0 <= col < resistance_surface.shape[1]:
                gps_coords.append((row, col))
        
        if len(gps_coords) < 2:
            print(f"   ⚠️  Insufficient GPS points within raster bounds")
            return np.zeros_like(resistance_surface)
        
        # Create cost-distance surface from all GPS points
        corridor_surface = np.zeros_like(resistance_surface)
        
        # Use a simplified approach: create corridors between sequential GPS points
        for i in range(len(gps_coords) - 1):
            start = gps_coords[i]
            end = gps_coords[i + 1]
            
            # Simple linear interpolation for corridor path
            path_length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
            
            if path_length > 0:
                for j in range(path_length + 1):
                    t = j / path_length if path_length > 0 else 0
                    row = int(start[0] + t * (end[0] - start[0]))
                    col = int(start[1] + t * (end[1] - start[1]))
                    
                    if 0 <= row < corridor_surface.shape[0] and 0 <= col < corridor_surface.shape[1]:
                        # Weight by inverse resistance
                        corridor_surface[row, col] += 1 / (resistance_surface[row, col] + 0.01)
        
        # Apply Gaussian filter to create corridors of realistic width
        corridor_surface = ndimage.gaussian_filter(corridor_surface, sigma=3)
        
        # Normalize
        if corridor_surface.max() > 0:
            corridor_surface = corridor_surface / corridor_surface.max()
        
        # Save corridor prediction
        self._save_raster(corridor_surface, f'{model_name.lower()}_corridors.tif',
                         f'{model_name} Corridor Prediction (0-1)')
        
        return corridor_surface
    
    def validate_models(self):
        """Validate both models against GPS tracking data."""
        print(f"\n✅ Validating Models Against GPS Data...")
        
        # Get GPS points in raster coordinates
        gps_utm = self.gps_data.to_crs('EPSG:32633')
        transform_obj = self.raster_profile['transform']
        
        gps_raster_coords = []
        for point in gps_utm.geometry:
            col, row = ~transform_obj * (point.x, point.y)
            col, row = int(col), int(row)
            
            if (0 <= row < self.raster_profile['height'] and 
                0 <= col < self.raster_profile['width']):
                gps_raster_coords.append((row, col))
        
        # Validation metrics for both models
        validation_results = {}
        
        for model_name, corridor_surface in self.corridor_predictions.items():
            
            # 1. Corridor overlap with GPS points
            gps_corridor_values = []
            for row, col in gps_raster_coords:
                gps_corridor_values.append(corridor_surface[row, col])
            
            mean_corridor_value = np.mean(gps_corridor_values)
            
            # 2. Corridor width analysis
            corridor_binary = corridor_surface > np.percentile(corridor_surface, 90)
            total_corridor_pixels = np.sum(corridor_binary)
            corridor_area_km2 = total_corridor_pixels * (self.raster_resolution / 1000) ** 2
            
            # 3. GPS point coverage
            covered_gps_points = sum(1 for val in gps_corridor_values if val > 0.1)
            coverage_percentage = (covered_gps_points / len(gps_corridor_values)) * 100
            
            # 4. Distance from corridors
            if total_corridor_pixels > 0:
                # Distance transform from corridor pixels
                distances = ndimage.distance_transform_edt(~corridor_binary) * self.raster_resolution
                
                gps_distances = []
                for row, col in gps_raster_coords:
                    gps_distances.append(distances[row, col])
                
                mean_distance = np.mean(gps_distances)
                median_distance = np.median(gps_distances)
            else:
                mean_distance = float('inf')
                median_distance = float('inf')
            
            # 5. Resistance values at GPS locations
            resistance_surface = self.raster_layers[model_name.lower()]
            gps_resistance_values = []
            for row, col in gps_raster_coords:
                gps_resistance_values.append(resistance_surface[row, col])
            
            mean_resistance = np.mean(gps_resistance_values)
            
            validation_results[model_name] = {
                'mean_corridor_value': mean_corridor_value,
                'corridor_area_km2': corridor_area_km2,
                'gps_coverage_percentage': coverage_percentage,
                'mean_distance_to_corridor_m': mean_distance,
                'median_distance_to_corridor_m': median_distance,
                'mean_resistance_at_gps': mean_resistance,
                'total_gps_points': len(gps_raster_coords)
            }
            
            print(f"   📊 {model_name} Model:")
            print(f"      GPS Coverage: {coverage_percentage:.1f}%")
            print(f"      Mean Distance to Corridor: {mean_distance:.0f}m")
            print(f"      Corridor Area: {corridor_area_km2:.1f} km²")
        
        # Compare models
        print("Available keys:", list(validation_results.keys()))
        energyscape_metrics = validation_results['energyscape']  # This will show you the actual keys
        traditional_metrics = validation_results['traditional']
        
        # Determine which model performs better
        energyscape_score = (
            energyscape_metrics['gps_coverage_percentage'] / 100 +
            (1 / (energyscape_metrics['mean_distance_to_corridor_m'] + 1)) * 1000 +
            energyscape_metrics['mean_corridor_value']
        ) / 3
        
        traditional_score = (
            traditional_metrics['gps_coverage_percentage'] / 100 +
            (1 / (traditional_metrics['mean_distance_to_corridor_m'] + 1)) * 1000 +
            traditional_metrics['mean_corridor_value']
        ) / 3
        
        comparison = {
            'energyscape_score': energyscape_score,
            'traditional_score': traditional_score,
            'better_model': 'Energyscape' if energyscape_score > traditional_score else 'Traditional',
            'score_difference': abs(energyscape_score - traditional_score)
        }
        
        validation_results['comparison'] = comparison
        self.validation_results = validation_results
        
        print(f"\n🎯 Model Comparison:")
        print(f"   Energyscape Score: {energyscape_score:.3f}")
        print(f"   Traditional Score: {traditional_score:.3f}")
        print(f"   Better Model: {comparison['better_model']}")
        
        # Save validation results
        with open(self.results_dir / 'validation_results.json', 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        return validation_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print(f"\n🎨 Creating Visualizations...")
        
        viz_dir = self.temp_dir / "visualizations"
        
        # 1. Raster Layers Overview
        self._create_raster_overview(viz_dir)
        
        # 2. Corridor Comparison
        self._create_corridor_comparison(viz_dir)
        
        # 3. Validation Analysis
        self._create_validation_plots(viz_dir)
        
        # 4. Summary Dashboard
        self._create_summary_dashboard(viz_dir)
        
        print(f"✅ Created visualizations in {viz_dir}")
    
    def _create_raster_overview(self, viz_dir):
        """Create overview of all raster layers."""
        
        # Select key rasters to display
        key_rasters = ['dem', 'slope', 'ndvi', 'water', 'human_pressure', 'resources']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, raster_name in enumerate(key_rasters):
            ax = axes[i]
            data = self.raster_layers[raster_name]
            
            # Choose appropriate colormap
            if raster_name == 'dem':
                cmap = 'terrain'
            elif raster_name == 'slope':
                cmap = 'Reds'
            elif raster_name == 'ndvi':
                cmap = 'Greens'
            elif raster_name == 'water':
                cmap = 'Blues'
            elif raster_name == 'human_pressure':
                cmap = 'Oranges'
            else:
                cmap = 'viridis'
            
            im = ax.imshow(data, cmap=cmap, aspect='equal')
            ax.set_title(f'{raster_name.upper()}', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.suptitle('Synthetic Environmental Raster Layers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'raster_layers_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_corridor_comparison(self, viz_dir):
        """Create side-by-side corridor comparison."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # GPS points for overlay
        gps_utm = self.gps_data.to_crs('EPSG:32633')
        transform_obj = self.raster_profile['transform']
        
        gps_x, gps_y = [], []
        for point in gps_utm.geometry:
            col, row = ~transform_obj * (point.x, point.y)
            gps_x.append(col)
            gps_y.append(row)
        
        # Plot Energyscape corridors
        ax1 = axes[0]
        im1 = ax1.imshow(self.corridor_predictions['energyscape'], cmap='Reds', alpha=0.8)
        ax1.scatter(gps_x, gps_y, c='blue', s=1, alpha=0.6, label='GPS Points')
        ax1.set_title('Energyscape Corridors', fontweight='bold', fontsize=14)
        ax1.legend()
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Plot Traditional corridors
        ax2 = axes[1]
        im2 = ax2.imshow(self.corridor_predictions['traditional'], cmap='Blues', alpha=0.8)
        ax2.scatter(gps_x, gps_y, c='red', s=1, alpha=0.6, label='GPS Points')
        ax2.set_title('Traditional Corridors', fontweight='bold', fontsize=14)
        ax2.legend()
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Plot Combined Comparison
        ax3 = axes[2]
        
        # Create RGB composite
        energyscape_norm = self.corridor_predictions['energyscape'] / self.corridor_predictions['energyscape'].max()
        traditional_norm = self.corridor_predictions['traditional'] / self.corridor_predictions['traditional'].max()
        
        rgb_array = np.zeros((*energyscape_norm.shape, 3))
        rgb_array[:, :, 0] = energyscape_norm  # Red channel for Energyscape
        rgb_array[:, :, 2] = traditional_norm  # Blue channel for Traditional
        rgb_array[:, :, 1] = (energyscape_norm + traditional_norm) / 2  # Green for overlap
        
        ax3.imshow(rgb_array, alpha=0.8)
        ax3.scatter(gps_x, gps_y, c='yellow', s=1, alpha=0.8, label='GPS Points')
        ax3.set_title('Corridor Comparison\n(Red=Energyscape, Blue=Traditional, Purple=Overlap)', 
                     fontweight='bold', fontsize=12)
        ax3.legend()
        
        # Remove axis labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Corridor Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'corridor_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_validation_plots(self, viz_dir):
        """Create validation analysis plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Data for plotting
        models = ['energyscape', 'traditional']
        coverage = [self.validation_results[model]['gps_coverage_percentage'] for model in models]
        distances = [self.validation_results[model]['mean_distance_to_corridor_m'] for model in models]
        areas = [self.validation_results[model]['corridor_area_km2'] for model in models]
        corridor_values = [self.validation_results[model]['mean_corridor_value'] for model in models]
        
        # GPS Coverage Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, coverage, color=['red', 'blue'], alpha=0.7)
        ax1.set_ylabel('GPS Coverage (%)')
        ax1.set_title('GPS Point Coverage by Corridors', fontweight='bold')
        ax1.set_ylim(0, 100)
        
        # Add value labels
        for bar, val in zip(bars1, coverage):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontweight='bold')
        
        # Distance to Corridors
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, distances, color=['red', 'blue'], alpha=0.7)
        ax2.set_ylabel('Mean Distance (m)')
        ax2.set_title('Mean Distance from GPS to Corridors', fontweight='bold')
        
        for bar, val in zip(bars2, distances):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(distances)*0.02,
                    f'{val:.0f}m', ha='center', fontweight='bold')
        
        # Corridor Areas
        ax3 = axes[1, 0]
        bars3 = ax3.bar(models, areas, color=['red', 'blue'], alpha=0.7)
        ax3.set_ylabel('Corridor Area (km²)')
        ax3.set_title('Total Corridor Area', fontweight='bold')
        
        for bar, val in zip(bars3, areas):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(areas)*0.02,
                    f'{val:.1f}', ha='center', fontweight='bold')
        
        # Model Scores
        ax4 = axes[1, 1]
        scores = [self.validation_results['comparison']['energyscape_score'],
                 self.validation_results['comparison']['traditional_score']]
        bars4 = ax4.bar(models, scores, color=['red', 'blue'], alpha=0.7)
        ax4.set_ylabel('Composite Score')
        ax4.set_title('Model Performance Scores', fontweight='bold')
        
        # Highlight better model
        better_model = self.validation_results['comparison']['better_model']
        for i, (bar, val, model) in enumerate(zip(bars4, scores, models)):
            color = 'gold' if model == better_model else 'black'
            weight = 'bold' if model == better_model else 'normal'
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scores)*0.02,
                    f'{val:.3f}', ha='center', fontweight=weight, color=color)
        
        plt.suptitle('Model Validation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_dashboard(self, viz_dir):
        """Create comprehensive summary dashboard."""
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Elephant Corridor Prediction Analysis - Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Study Area Overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Show DEM with GPS points
        dem_display = ax1.imshow(self.raster_layers['dem'], cmap='terrain', alpha=0.8)
        
        # Overlay GPS track
        gps_utm = self.gps_data.to_crs('EPSG:32633')
        transform_obj = self.raster_profile['transform']
        
        gps_x, gps_y = [], []
        for point in gps_utm.geometry:
            col, row = ~transform_obj * (point.x, point.y)
            gps_x.append(col)
            gps_y.append(row)
        print("GPS X:", gps_x)
        print("GPS Y:", gps_y)
        ax1.plot(gps_x, gps_y, 'red', linewidth=2, alpha=0.8, label='GPS Track')
        ax1.scatter(gps_x[0], gps_y[0], c='green', s=100, marker='o', 
                   edgecolors='white', linewidth=2, label='Start', zorder=5)
        ax1.scatter(gps_x[-1], gps_y[-1], c='red', s=100, marker='s', 
                   edgecolors='white', linewidth=2, label='End', zorder=5)
        
        ax1.set_title('Study Area & GPS Tracking Data', fontweight='bold')
        ax1.legend()
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(dem_display, ax=ax1, shrink=0.6)
        
        # 2. Model Comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        models = ['energyscape', 'traditional']

        coverage = [self.validation_results[model]['gps_coverage_percentage'] for model in models]
        
        bars = ax2.bar(models, coverage, color=['darkred', 'darkblue'], alpha=0.8)
        ax2.set_ylabel('GPS Coverage (%)')
        ax2.set_title('Model Performance Comparison', fontweight='bold')
        ax2.set_ylim(0, 100)
        
        # Highlight better model
        better_model = self.validation_results['comparison']['better_model']
        for bar, val, model in zip(bars, coverage, models):
            color = 'gold' if model == better_model else 'white'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', fontweight='bold', color=color)
        
        # 3. Corridor Predictions (middle row)
        ax3 = fig.add_subplot(gs[1, :2])
        im3 = ax3.imshow(self.corridor_predictions['energyscape'], cmap='Reds', alpha=0.8)
        ax3.scatter(gps_x, gps_y, c='blue', s=5, alpha=0.7)
        ax3.set_title('Energyscape Corridors', fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.colorbar(im3, ax=ax3, shrink=0.6)
        
        ax4 = fig.add_subplot(gs[1, 2:])
        im4 = ax4.imshow(self.corridor_predictions['traditional'], cmap='Blues', alpha=0.8)
        ax4.scatter(gps_x, gps_y, c='red', s=5, alpha=0.7)
        ax4.set_title('Traditional Corridors', fontweight='bold')
        ax4.set_xticks([])
        ax4.set_yticks([])
        plt.colorbar(im4, ax=ax4, shrink=0.6)
        
        # 4. Statistics Summary (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create statistics table
        _, study_duration_text = self._get_study_duration_info()

        stats_text = f"""
ANALYSIS SUMMARY

GPS Tracking Data:
• Total GPS Points: {len(self.gps_data)}
• Study Duration: {study_duration_text}
• AOI Area: {self.aoi_area_km2:.1f} km²"""

        ax5.text(0.5, 0.5, stats_text.strip(), transform=ax5.transAxes,
                fontsize=11, fontfamily='monospace', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.savefig(viz_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_final_report(self):
        """Generate final comprehensive report."""
        print(f"\n📋 Generating Final Report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        study_duration_days, study_duration_text = self._get_study_duration_info()
        # Use a default if study_duration_days is None (e.g., "N/A" or 0)
        study_duration_days_for_json = study_duration_days if study_duration_days is not None else "N/A"

        report = {
            'analysis_metadata': {
                'timestamp': timestamp,
                'analyst': 'Ecological AI Research Assistant',
                'study_region': 'Cameroon Forest Elephant Habitat',
                'analysis_type': 'Corridor Prediction Model Comparison'
            },
            'data_summary': {
                'gps_points': len(self.gps_data),
                'study_duration_days': study_duration_days_for_json,
                'raster_resolution_m': self.raster_resolution,
                'study_area_km2': (self.raster_profile['width'] * self.raster_profile['height'] * 
                                 (self.raster_resolution / 1000) ** 2),
                'environmental_layers': len(self.raster_layers)
            },
            'model_parameters': {
                'energyscape': {
                    'elephant_mass_kg': self.elephant_mass_kg,
                    'factors': ['elevation', 'slope', 'vegetation', 'water_availability'],
                    'energy_cost_calculation': 'terrain-based metabolic cost'
                },
                'traditional': {
                    'factors': ['land_cover', 'human_pressure', 'slope', 'water', 'vegetation', 'risk'],
                    'resistance_calculation': 'weighted multi-factor resistance'
                }
            },
            'validation_results': self.validation_results,
            'conclusions': {
                'better_model': self.validation_results['comparison']['better_model'],
                'performance_difference': f"{self.validation_results['comparison']['score_difference']:.3f}",
                'key_findings': [
                    f"GPS Coverage: Energyscape {self.validation_results['energyscape']['gps_coverage_percentage']:.1f}% vs Traditional {self.validation_results['traditional']['gps_coverage_percentage']:.1f}%",
                    f"Distance Accuracy: Energyscape {self.validation_results['energyscape']['mean_distance_to_corridor_m']:.0f}m vs Traditional {self.validation_results['traditional']['mean_distance_to_corridor_m']:.0f}m",
                    f"The {self.validation_results['comparison']['better_model']} model shows superior performance for this dataset"
                ]
            },
            'files_generated': {
                'raster_layers': [str(p) for p in self.raster_dir.glob('*.tif')],
                'corridor_predictions': [str(p) for p in (self.temp_dir / 'corridors').glob('*.tif')],
                'visualizations': [str(p) for p in (self.temp_dir / 'visualizations').glob('*.png')],
                'results': [str(p) for p in self.results_dir.glob('*.json')]
            }
        }
        
        # Save comprehensive report
        report_file = self.results_dir / f'corridor_analysis_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str) # Using default=str for Path objects
        
        # Create summary table for easy reading
        summary_file = self.results_dir / f'model_comparison_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write("ELEPHANT CORRIDOR PREDICTION ANALYSIS - MODEL COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("STUDY OVERVIEW:\n")
            f.write(f"• GPS Points Analyzed: {len(self.gps_data):,}\n")
            f.write(f"• Study Duration: {study_duration_text}\n") # Use text from helper
            f.write(f"• Analysis Resolution: {self.raster_resolution}m\n")
            f.write(f"• Environmental Layers: {len(self.raster_layers)}\n\n")
            
            f.write("MODEL PERFORMANCE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Metric':<25} {'Energyscape':<15} {'Traditional':<15}\n")
            f.write("-" * 40 + "\n")
            
            metrics = [
                ('GPS Coverage (%)', 'gps_coverage_percentage', '.1f'),
                ('Mean Distance (m)', 'mean_distance_to_corridor_m', '.0f'),
                ('Corridor Area (km²)', 'corridor_area_km2', '.1f'),
                ('Mean Corridor Value', 'mean_corridor_value', '.3f')
            ]
            
            for metric_name, key, fmt in metrics:
                # Use .get() for safer dictionary access, though keys should exist
                e_val = self.validation_results.get('energyscape', {}).get(key, 'N/A')
                t_val = self.validation_results.get('traditional', {}).get(key, 'N/A')
                
                # Handle N/A for formatting
                e_val_str = f"{e_val:{fmt}}" if isinstance(e_val, (int, float)) else str(e_val)
                t_val_str = f"{t_val:{fmt}}" if isinstance(t_val, (int, float)) else str(t_val)
                f.write(f"{metric_name:<25} {e_val_str:<15} {t_val_str:<15}\n")

            f.write("\nOVERALL PERFORMANCE:\n")
            f.write(f"• Best Model: {self.validation_results['comparison']['better_model']}\n")
            f.write(f"• Performance Difference: {self.validation_results['comparison']['score_difference']:.3f}\n")
            
            f.write("\nKEY FINDINGS:\n")
            for finding in report['conclusions']['key_findings']:
                f.write(f"• {finding}\n")
        
        print(f"✅ Final report generated:")
        print(f"   📄 Comprehensive: {report_file}")
        print(f"   📄 Summary: {summary_file}")
        
        return report
    
    def run_complete_analysis(self):
        """Run the complete corridor prediction analysis workflow."""
        
        print(f"🚀 STARTING COMPREHENSIVE CORRIDOR PREDICTION ANALYSIS")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Data Preparation
            self.load_step2_data()
            
            # Step 2: Synthetic Raster Generation
            self.generate_synthetic_rasters()
            
            # Step 3: Energyscape Model
            self.calculate_energyscape_model()
            
            # Step 4: Traditional Model
            self.calculate_traditional_model()
            
            # Step 5: Model Validation
            self.validate_models()
            
            # Step 6: Visualizations
            self.create_visualizations()
            
            # Step 7: Final Report
            final_report = self.generate_final_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"⏱️  Total Duration: {duration}")
            print(f"📁 Output Directory: {self.temp_dir}")
            print(f"🏆 Best Model: {self.validation_results['comparison']['better_model']}")
            print(f"📊 Performance Difference: {self.validation_results['comparison']['score_difference']:.3f}")
            
            print(f"\n📂 Generated Files:")
            print(f"   🗺️  Raster Layers: {len(list(self.raster_dir.glob('*.tif')))} files")
            print(f"   🛤️  Corridor Predictions: 2 models")
            print(f"   📊 Visualizations: {len(list((self.temp_dir / 'visualizations').glob('*.png')))} plots")
            print(f"   📋 Reports: {len(list(self.results_dir.glob('*')))} files")
            
            return final_report
            
        except Exception as e:
            print(f"\n❌ ANALYSIS FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the corridor prediction analysis."""
    
    print("🐘 ELEPHANT CORRIDOR PREDICTION ANALYSIS")
    print("Comparing Energyscape vs Traditional Raster-based Models")
    print("=" * 80)
    
    # Initialize analysis
    # Update this path to point to your Step 2 directory
    step2_path = "../STEP 2"  # Adjust this path as needed
    
    analyzer = CorridorPredictionAnalysis(step2_data_path=step2_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Check '{TEMP_DATA_DIR}' for all results and visualizations")
        print(f"🎯 Use the generated files for your research presentation")
    else:
        print(f"\n❌ Analysis failed - check error messages above")

if __name__ == "__main__":
    main()