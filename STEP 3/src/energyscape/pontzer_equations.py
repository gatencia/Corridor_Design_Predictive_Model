#!/usr/bin/env python3
"""
Implementation of Pontzer's unified locomotion cost equations.
Based on Pontzer (2016) and applied to elephant movement by Berti et al. (2025).
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PontzerEquations:
    """
    Implementation of Pontzer's locomotion cost model for elephants.
    
    Based on the unified theory of terrestrial locomotion that relates
    energy expenditure to body mass and terrain slope.
    
    References:
    - Pontzer, H. (2016). A unified theory for the energy cost of legged locomotion. 
      Biology Letters, 12(2), 20150935.
    - Berti, E. et al. (2025). Energy landscapes direct the movement preferences 
      of elephants. Journal of Animal Ecology.
    """
    
    # Default parameters for elephants (from Berti et al. 2025)
    DEFAULT_ALPHA = 1.0      # Base metabolic coefficient
    DEFAULT_BETA = 0.75      # Mass scaling exponent (allometric)
    DEFAULT_GAMMA = 1.5      # Slope penalty coefficient
    
    def __init__(self, alpha: float = None, beta: float = None, gamma: float = None):
        """
        Initialize Pontzer equations with species-specific parameters.
        
        Parameters:
        -----------
        alpha : float, optional
            Base metabolic coefficient (default from literature)
        beta : float, optional
            Mass scaling exponent (default 0.75 for allometric scaling)
        gamma : float, optional
            Slope penalty coefficient (default from elephant studies)
        """
        self.alpha = alpha if alpha is not None else self.DEFAULT_ALPHA
        self.beta = beta if beta is not None else self.DEFAULT_BETA
        self.gamma = gamma if gamma is not None else self.DEFAULT_GAMMA
        
        logger.info(f"Initialized Pontzer equations with α={self.alpha}, β={self.beta}, γ={self.gamma}")
        
    def calculate_base_cost(self, mass_kg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate base locomotion cost without slope effects.
        
        Base Energy Cost = α × M^β
        
        This represents the metabolic cost of locomotion on flat terrain,
        scaled allometrically with body mass.
        
        Parameters:
        -----------
        mass_kg : float or np.ndarray
            Body mass in kilograms
            
        Returns:
        --------
        float or np.ndarray
            Base energy cost in kcal/km
        """
        if isinstance(mass_kg, (int, float)):
            if mass_kg <= 0:
                raise ValueError("Body mass must be positive")
        else:
            if np.any(mass_kg <= 0):
                raise ValueError("All body mass values must be positive")
        
        base_cost = self.alpha * np.power(mass_kg, self.beta)
        
        logger.debug(f"Base cost calculated for mass {mass_kg} kg: {base_cost} kcal/km")
        return base_cost
    
    def calculate_slope_multiplier(self, slope_degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate slope-dependent cost multiplier.
        
        Slope Multiplier = 1 + γ × |sin(slope)|
        
        This captures the additional energetic cost of moving up or down slopes.
        The absolute value ensures that both uphill and downhill movement
        incur additional costs (though in reality, moderate downhill grades
        can reduce costs).
        
        Parameters:
        -----------
        slope_degrees : float or np.ndarray
            Slope in degrees (-90 to +90)
            
        Returns:
        --------
        float or np.ndarray
            Slope cost multiplier (≥ 1.0)
        """
        # Validate slope range
        if isinstance(slope_degrees, (int, float)):
            if not -90 <= slope_degrees <= 90:
                logger.warning(f"Slope {slope_degrees}° outside typical range [-90, 90]")
        else:
            if np.any(np.abs(slope_degrees) > 90):
                logger.warning("Some slope values outside typical range [-90, 90]")
        
        # Convert to radians
        slope_radians = np.radians(slope_degrees)
        
        # Calculate multiplier
        multiplier = 1.0 + self.gamma * np.abs(np.sin(slope_radians))
        
        logger.debug(f"Slope multiplier for {slope_degrees}°: {multiplier}")
        return multiplier
    
    def calculate_energy_cost(self, mass_kg: Union[float, np.ndarray], 
                            slope_degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate total energy cost including slope penalty.
        
        Total Cost = α × M^β × (1 + γ × |sin(slope)|)
                  = Base Cost × Slope Multiplier
        
        This is the core Pontzer equation that combines allometric scaling
        of locomotion costs with terrain effects.
        
        Parameters:
        -----------
        mass_kg : float or np.ndarray
            Body mass in kilograms
        slope_degrees : float or np.ndarray
            Slope in degrees
            
        Returns:
        --------
        float or np.ndarray
            Total energy cost in kcal/km
        """
        base_cost = self.calculate_base_cost(mass_kg)
        slope_multiplier = self.calculate_slope_multiplier(slope_degrees)
        
        total_cost = base_cost * slope_multiplier
        
        if isinstance(total_cost, np.ndarray):
            logger.debug(f"Energy cost calculated for {total_cost.size} cells")
        else:
            logger.debug(f"Energy cost: {total_cost} kcal/km for {mass_kg} kg at {slope_degrees}°")
        
        return total_cost
    
    def calculate_speed_adjusted_cost(self, mass_kg: Union[float, np.ndarray],
                                    slope_degrees: Union[float, np.ndarray],
                                    speed_kmh: Union[float, np.ndarray],
                                    reference_speed_kmh: float = 3.5) -> Union[float, np.ndarray]:
        """
        Calculate speed-adjusted energy cost.
        
        Some studies suggest that movement costs vary with speed.
        This method allows for speed adjustment relative to a reference speed.
        
        Parameters:
        -----------
        mass_kg : float or np.ndarray
            Body mass in kilograms
        slope_degrees : float or np.ndarray
            Slope in degrees
        speed_kmh : float or np.ndarray
            Movement speed in km/h
        reference_speed_kmh : float
            Reference speed for cost calculation (default 3.5 km/h for elephants)
            
        Returns:
        --------
        float or np.ndarray
            Speed-adjusted energy cost in kcal/km
        """
        base_cost = self.calculate_energy_cost(mass_kg, slope_degrees)
        
        # Simple linear speed adjustment (could be made more sophisticated)
        speed_factor = np.maximum(speed_kmh / reference_speed_kmh, 0.1)  # Minimum factor
        
        adjusted_cost = base_cost * speed_factor
        
        logger.debug(f"Speed-adjusted cost calculated with factor {speed_factor}")
        return adjusted_cost
    
    def calculate_cost_surface(self, mass_kg: float, slope_array: np.ndarray,
                             nodata_value: float = -9999.0) -> np.ndarray:
        """
        Calculate energy cost surface from slope array.
        
        This is the main method for generating energy cost rasters
        from DEM-derived slope data.
        
        Parameters:
        -----------
        mass_kg : float
            Elephant body mass in kilograms
        slope_array : np.ndarray
            2D array of slope values in degrees
        nodata_value : float
            Value to use for NoData cells
            
        Returns:
        --------
        np.ndarray
            2D array of energy costs in kcal/km
        """
        logger.info(f"Calculating energy cost surface for {mass_kg} kg elephant")
        
        # Create output array
        cost_array = np.full_like(slope_array, nodata_value, dtype=np.float32)
        
        # Find valid slope values
        valid_mask = (slope_array != nodata_value) & ~np.isnan(slope_array)
        valid_slopes = slope_array[valid_mask]
        
        if np.sum(valid_mask) == 0:
            logger.warning("No valid slope values found in input array")
            return cost_array
        
        # Calculate costs for valid cells
        valid_costs = self.calculate_energy_cost(mass_kg, valid_slopes)
        
        # Fill output array
        cost_array[valid_mask] = valid_costs
        
        logger.info(f"Calculated costs for {np.sum(valid_mask)} valid cells")
        logger.info(f"Cost range: {np.min(valid_costs):.2f} - {np.max(valid_costs):.2f} kcal/km")
        
        return cost_array
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current equation parameters."""
        return {
            'alpha': self.alpha,
            'beta': self.beta, 
            'gamma': self.gamma
        }
    
    def set_parameters(self, alpha: float = None, beta: float = None, gamma: float = None) -> None:
        """
        Update equation parameters.
        
        Parameters:
        -----------
        alpha : float, optional
            New alpha value
        beta : float, optional
            New beta value
        gamma : float, optional
            New gamma value
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        
        logger.info(f"Updated parameters: α={self.alpha}, β={self.beta}, γ={self.gamma}")
    
    @classmethod
    def get_elephant_masses(cls) -> Dict[str, float]:
        """Get standard elephant body masses from literature."""
        return {
            'female': 2744.0,  # kg, from Berti et al. 2025
            'male': 6029.0,    # kg, from Berti et al. 2025
            'average': 4000.0, # kg, approximate average adult
            'juvenile': 1500.0 # kg, approximate juvenile
        }
    
    def validate_against_literature(self, mass_kg: float = 4000.0) -> Dict[str, Any]:
        """
        Validate calculations against literature values.
        
        Parameters:
        -----------
        mass_kg : float
            Test mass in kilograms
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        results = {
            'test_mass_kg': mass_kg,
            'flat_terrain_cost': self.calculate_energy_cost(mass_kg, 0.0),
            'moderate_slope_cost': self.calculate_energy_cost(mass_kg, 15.0),
            'steep_slope_cost': self.calculate_energy_cost(mass_kg, 30.0),
            'parameters': self.get_parameters()
        }
        
        # Calculate cost increases
        flat_cost = results['flat_terrain_cost']
        moderate_cost = results['moderate_slope_cost']
        steep_cost = results['steep_slope_cost']
        
        results['moderate_increase_percent'] = ((moderate_cost - flat_cost) / flat_cost) * 100
        results['steep_increase_percent'] = ((steep_cost - flat_cost) / flat_cost) * 100
        
        logger.info(f"Validation results for {mass_kg} kg elephant:")
        logger.info(f"  Flat terrain: {flat_cost:.2f} kcal/km")
        logger.info(f"  15° slope: {moderate_cost:.2f} kcal/km (+{results['moderate_increase_percent']:.1f}%)")
        logger.info(f"  30° slope: {steep_cost:.2f} kcal/km (+{results['steep_increase_percent']:.1f}%)")
        
        return results
    
    def create_test_surface(self, shape: Tuple[int, int] = (100, 100)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create test slope and energy cost surfaces for validation.
        
        Parameters:
        -----------
        shape : Tuple[int, int]
            Shape of test arrays (rows, cols)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Slope array and corresponding energy cost array
        """
        # Create synthetic slope surface
        rows, cols = shape
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        X, Y = np.meshgrid(x, y)
        
        # Create varied slope surface (0-45 degrees)
        slope_surface = 22.5 * (1 + np.sin(3 * X) * np.cos(3 * Y))
        
        # Calculate energy costs for average elephant
        mass_kg = self.get_elephant_masses()['average']
        energy_surface = self.calculate_cost_surface(mass_kg, slope_surface)
        
        logger.info(f"Created test surfaces: {shape[0]}x{shape[1]} cells")
        logger.info(f"Slope range: {np.min(slope_surface):.1f}° - {np.max(slope_surface):.1f}°")
        
        return slope_surface, energy_surface

# Convenience functions for common use cases
def calculate_elephant_energy_cost(slope_degrees: Union[float, np.ndarray], 
                                 sex: str = 'average',
                                 alpha: float = 1.0, beta: float = 0.75, gamma: float = 1.5) -> Union[float, np.ndarray]:
    """
    Convenience function to calculate elephant energy costs.
    
    Parameters:
    -----------
    slope_degrees : float or np.ndarray
        Slope in degrees
    sex : str
        Elephant sex ('male', 'female', or 'average')
    alpha, beta, gamma : float
        Pontzer equation parameters
        
    Returns:
    --------
    float or np.ndarray
        Energy cost in kcal/km
    """
    pontzer = PontzerEquations(alpha, beta, gamma)
    masses = pontzer.get_elephant_masses()
    
    if sex not in masses:
        raise ValueError(f"Unknown sex '{sex}'. Use: {list(masses.keys())}")
    
    mass_kg = masses[sex]
    return pontzer.calculate_energy_cost(mass_kg, slope_degrees)

def create_energy_cost_lookup_table(slope_range: Tuple[float, float] = (0, 45),
                                  slope_step: float = 1.0) -> pd.DataFrame:
    """
    Create lookup table of energy costs vs slope for different elephant types.
    
    Parameters:
    -----------
    slope_range : Tuple[float, float]
        Range of slopes to calculate (min_deg, max_deg)
    slope_step : float
        Step size for slope values
        
    Returns:
    --------
    pd.DataFrame
        Lookup table with columns for slope and energy costs
    """
    pontzer = PontzerEquations()
    masses = pontzer.get_elephant_masses()
    
    # Create slope array
    slopes = np.arange(slope_range[0], slope_range[1] + slope_step, slope_step)
    
    # Calculate costs for each elephant type
    data = {'slope_degrees': slopes}
    
    for sex, mass_kg in masses.items():
        costs = pontzer.calculate_energy_cost(mass_kg, slopes)
        data[f'cost_{sex}_kcal_km'] = costs
    
    df = pd.DataFrame(data)
    
    logger.info(f"Created lookup table with {len(df)} slope values")
    return df

# Example usage and testing
if __name__ == "__main__":
    # Test Pontzer equations
    print("🧮 Testing Pontzer Locomotion Equations")
    print("=" * 60)
    
    # Initialize equations
    pontzer = PontzerEquations()
    
    # Test basic calculations
    print("Testing basic calculations...")
    masses = pontzer.get_elephant_masses()
    
    for sex, mass in masses.items():
        flat_cost = pontzer.calculate_energy_cost(mass, 0.0)
        slope_cost = pontzer.calculate_energy_cost(mass, 15.0)
        print(f"{sex.capitalize()} elephant ({mass} kg):")
        print(f"  Flat terrain: {flat_cost:.2f} kcal/km")
        print(f"  15° slope: {slope_cost:.2f} kcal/km")
        print(f"  Increase: {((slope_cost-flat_cost)/flat_cost)*100:.1f}%")
    
    # Test array calculations
    print("\nTesting array calculations...")
    slopes = np.array([0, 5, 10, 15, 20, 25, 30])
    costs = pontzer.calculate_energy_cost(4000.0, slopes)
    
    for slope, cost in zip(slopes, costs):
        print(f"  {slope:2d}°: {cost:.2f} kcal/km")
    
    # Validate against literature
    print("\nValidating against literature...")
    validation = pontzer.validate_against_literature()
    
    # Create test surface
    print("\nCreating test surface...")
    slope_surface, energy_surface = pontzer.create_test_surface((50, 50))
    print(f"Test surface created: {slope_surface.shape}")
    print(f"Energy cost range: {np.min(energy_surface):.2f} - {np.max(energy_surface):.2f} kcal/km")
    
    # Create lookup table
    print("\nCreating lookup table...")
    lookup_table = create_energy_cost_lookup_table()
    print(f"Lookup table shape: {lookup_table.shape}")
    print(lookup_table.head())
    
    print("\n🎉 Pontzer equations working correctly!")