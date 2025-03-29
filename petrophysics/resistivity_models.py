"""
Petrophysical models for relating water content/saturation to electrical resistivity.
"""
import numpy as np
from scipy.optimize import fsolve
from typing import Union, Optional, Tuple, List, Dict


class BaseResistivityModel:
    """Base class for resistivity models."""
    
    def __init__(self):
        """Initialize base resistivity model."""
        pass
    
    def calculate_resistivity(self, saturation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate resistivity from saturation.
        
        Args:
            saturation: Water saturation values
            **kwargs: Additional parameters for specific models
            
        Returns:
            Resistivity values
        """
        raise NotImplementedError("Resistivity calculation must be implemented in derived classes")
    
    def estimate_saturation(self, resistivity: np.ndarray, **kwargs) -> np.ndarray:
        """
        Estimate saturation from resistivity.
        
        Args:
            resistivity: Resistivity values
            **kwargs: Additional parameters for specific models
            
        Returns:
            Estimated saturation values
        """
        raise NotImplementedError("Saturation estimation must be implemented in derived classes")


class ArchieModel(BaseResistivityModel):
    """Archie's law for relating saturation to resistivity."""
    
    def __init__(self, a: float = 1.0, m: float = 2.0, n: float = 2.0):
        """
        Initialize Archie's law model.
        
        Args:
            a: Tortuosity factor
            m: Cementation exponent
            n: Saturation exponent
        """
        super().__init__()
        self.a = a
        self.m = m
        self.n = n
    
    def calculate_resistivity(self, saturation: np.ndarray, 
                            fluid_resistivity: Union[float, np.ndarray],
                            porosity: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate resistivity using Archie's law.
        
        Args:
            saturation: Water saturation values
            fluid_resistivity: Resistivity of the formation fluid
            porosity: Porosity values
            
        Returns:
            Formation resistivity values
        """
        # Formation factor
        F = self.a / (porosity ** self.m)
        
        # Resistivity at full saturation
        R_o = F * fluid_resistivity
        
        # Resistivity at partial saturation
        resistivity = R_o / (saturation ** self.n)
        
        return resistivity
    
    def estimate_saturation(self, resistivity: np.ndarray,
                          fluid_resistivity: Union[float, np.ndarray],
                          porosity: Union[float, np.ndarray]) -> np.ndarray:
        """
        Estimate saturation from resistivity using Archie's law.
        
        Args:
            resistivity: Formation resistivity values
            fluid_resistivity: Resistivity of the formation fluid
            porosity: Porosity values
            
        Returns:
            Estimated water saturation values
        """
        # Formation factor
        F = self.a / (porosity ** self.m)
        
        # Resistivity at full saturation
        R_o = F * fluid_resistivity
        
        # Saturation calculation
        saturation = (R_o / resistivity) ** (1 / self.n)
        
        # Ensure saturation is between 0 and 1
        saturation = np.clip(saturation, 0.0, 1.0)
        
        return saturation


class WaxmanSmitsModel(BaseResistivityModel):
    """Waxman-Smits model accounting for clay conductivity effects."""
    
    def __init__(self, n: float = 2.0, B: float = 1.0):
        """
        Initialize Waxman-Smits model.
        
        Args:
            n: Saturation exponent
            B: Equivalent counterion conductance (m²/(S·meq))
        """
        super().__init__()
        self.n = n
        self.B = B
    
    def calculate_resistivity(self, saturation: np.ndarray, 
                            base_resistivity: Union[float, np.ndarray],
                            Qv: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate resistivity using the Waxman-Smits model.
        
        Args:
            saturation: Water saturation values
            base_resistivity: Base resistivity at full saturation without clay effects
            Qv: Cation exchange capacity per unit pore volume (meq/ml)
            
        Returns:
            Formation resistivity values
        """
        # Surface conductivity factor
        BQv = self.B * Qv
        
        # Waxman-Smits equation
        resistivity = base_resistivity / (saturation ** self.n * (1 + base_resistivity * BQv * saturation ** (self.n - 1)))
        
        return resistivity
    
    def estimate_saturation(self, resistivity: np.ndarray,
                          base_resistivity: Union[float, np.ndarray],
                          Qv: Union[float, np.ndarray]) -> np.ndarray:
        """
        Estimate saturation from resistivity using the Waxman-Smits model.
        
        Args:
            resistivity: Formation resistivity values
            base_resistivity: Base resistivity at full saturation without clay effects
            Qv: Cation exchange capacity per unit pore volume (meq/ml)
            
        Returns:
            Estimated water saturation values
        """
        # Surface conductivity factor
        BQv = self.B * Qv
        
        # Create a function to solve for saturation
        def equation_to_solve(S, rho, R_o, BQv, n):
            return (R_o / S ** n) / (1 + R_o * BQv * S ** (n - 1)) - rho
        
        # Initialize saturation estimate using Archie's law
        S_initial = (base_resistivity / resistivity) ** (1 / self.n)
        
        # Apply fsolve to each resistivity value
        saturation = np.zeros_like(resistivity)
        
        for i in range(len(resistivity)):
            # Get scalar values for this element
            rho = resistivity[i] if np.isscalar(resistivity) else resistivity.flat[i]
            R_o = base_resistivity if np.isscalar(base_resistivity) else base_resistivity.flat[i]
            bqv = BQv if np.isscalar(BQv) else BQv.flat[i]
            S_init = S_initial if np.isscalar(S_initial) else S_initial.flat[i]
            
            # Bound S_init between 0.01 and 1.0 to ensure stability
            S_init = max(0.01, min(1.0, S_init))
            
            # Solve the equation
            solution = fsolve(equation_to_solve, S_init, args=(rho, R_o, bqv, self.n))
            
            # Store the result
            saturation.flat[i] = max(0.0, min(1.0, solution[0]))
        
        return saturation


class ModifiedWaxmanSmits(BaseResistivityModel):
    """Modified Waxman-Smits model with explicit surface conductivity term."""
    
    def __init__(self, n_model: Optional[Union[float, np.ndarray]] = None):
        """
        Initialize modified Waxman-Smits model.
        
        Args:
            n_model: Saturation exponent (can be array for different regions)
        """
        super().__init__()
        self.n_model = n_model if n_model is not None else 2.0
    
    def calculate_resistivity(self, saturation: np.ndarray,
                            base_resistivity: np.ndarray,
                            surface_conductivity: np.ndarray,
                            regions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate resistivity using modified Waxman-Smits equation.
        
        Args:
            saturation: Water saturation values
            base_resistivity: Base resistivity values
            surface_conductivity: Surface conductivity values
            regions: Region markers for different n values (if n_model is array)
            
        Returns:
            Formation resistivity values
        """
        # Initialize output array
        resistivity = np.zeros_like(saturation, dtype=float)
        
        # Handle uniform n case
        if np.isscalar(self.n_model):
            resistivity = base_resistivity * saturation ** (-self.n_model) + \
                         surface_conductivity * saturation ** (self.n_model - 1)
            return resistivity
        
        # Handle region-specific n values
        if regions is None:
            raise ValueError("Region markers must be provided when using region-specific n values")
        
        for i, n in enumerate(self.n_model):
            mask = (regions == i)
            resistivity[mask] = base_resistivity[mask] * saturation[mask] ** (-n) + \
                              surface_conductivity[mask] * saturation[mask] ** (n - 1)
        
        return resistivity
    
    def estimate_saturation(self, resistivity: np.ndarray,
                          base_resistivity: np.ndarray, 
                          surface_conductivity: np.ndarray,
                          regions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate saturation from resistivity using modified Waxman-Smits equation.
        
        Args:
            resistivity: Formation resistivity values
            base_resistivity: Base resistivity values
            surface_conductivity: Surface conductivity values
            regions: Region markers for different n values (if n_model is array)
            
        Returns:
            Estimated water saturation values
        """
        # Initialize saturation array
        saturation = np.zeros_like(resistivity)
        
        # Function to solve for saturation
        def equation_to_solve(S, rho, R_o, sigma_sur, n):
            return R_o * S ** (-n) + sigma_sur * S ** (n - 1) - rho
        
        # For scalar n case
        if np.isscalar(self.n_model):
            n = self.n_model
            
            # Generate initial guess using simple Archie's law
            S_initial = (base_resistivity / resistivity) ** (1 / n)
            S_initial = np.clip(S_initial, 0.01, 1.0)
            
            # Solve for each element
            for i in range(len(resistivity)):
                # Get scalar values for this element
                rho = resistivity.flat[i]
                R_o = base_resistivity.flat[i]
                sigma_sur = surface_conductivity.flat[i]
                S_init = S_initial.flat[i]
                
                # Solve equation
                solution = fsolve(equation_to_solve, S_init, args=(rho, R_o, sigma_sur, n))
                saturation.flat[i] = max(0.0, min(1.0, solution[0]))
            
            return saturation
        
        # For region-specific n case
        if regions is None:
            raise ValueError("Region markers must be provided when using region-specific n values")
        
        for i, n in enumerate(self.n_model):
            mask = (regions == i)
            
            # Skip if no cells in this region
            if not np.any(mask):
                continue
            
            # Extract values for this region
            rho_region = resistivity[mask]
            R_o_region = base_resistivity[mask]
            sigma_sur_region = surface_conductivity[mask]
            
            # Initial guess
            S_initial = (R_o_region / rho_region) ** (1 / n)
            S_initial = np.clip(S_initial, 0.01, 1.0)
            
            # Solve for each element in this region
            sat_region = np.zeros_like(rho_region)
            for j in range(len(rho_region)):
                solution = fsolve(equation_to_solve, S_initial[j], 
                                 args=(rho_region[j], R_o_region[j], sigma_sur_region[j], n))
                sat_region[j] = max(0.0, min(1.0, solution[0]))
            
            # Assign results back to the full array
            saturation[mask] = sat_region
        
        return saturation


class HybridResistivityModel(BaseResistivityModel):
    """
    Hybrid resistivity model combining elements from Archie's law, Waxman-Smits,
    and other empirical relationships with region-specific parameters.
    """
    
    def __init__(self, regions_config: Dict[int, Dict[str, Any]]):
        """
        Initialize hybrid resistivity model with region-specific configurations.
        
        Args:
            regions_config: Dictionary mapping region markers to parameter dictionaries
                Each region dictionary should contain:
                - 'model_type': str ('archie', 'waxman_smits', or 'modified_ws')
                - Model-specific parameters (a, m, n, etc.)
        """
        super().__init__()
        self.regions_config = regions_config
        
        # Create sub-models for each region
        self.region_models = {}
        for marker, config in regions_config.items():
            model_type = config.get('model_type', 'archie').lower()
            
            if model_type == 'archie':
                self.region_models[marker] = ArchieModel(
                    a=config.get('a', 1.0),
                    m=config.get('m', 2.0),
                    n=config.get('n', 2.0)
                )
            elif model_type == 'waxman_smits':
                self.region_models[marker] = WaxmanSmitsModel(
                    n=config.get('n', 2.0),
                    B=config.get('B', 1.0)
                )
            elif model_type == 'modified_ws':
                self.region_models[marker] = ModifiedWaxmanSmits(
                    n_model=config.get('n', 2.0)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
    
    def calculate_resistivity(self, saturation: np.ndarray, porosity: np.ndarray,
                            regions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate resistivity using region-specific models.
        
        Args:
            saturation: Water saturation values
            porosity: Porosity values
            regions: Region markers for each cell
            **kwargs: Additional parameters including:
                - fluid_resistivity: Resistivity of the formation fluid
                - base_resistivity: Base resistivity for Waxman-Smits
                - surface_conductivity: Surface conductivity for modified WS
                
        Returns:
            Formation resistivity values
        """
        resistivity = np.zeros_like(saturation, dtype=float)
        
        # Calculate resistivity for each region
        for marker, model in self.region_models.items():
            # Get mask for this region
            mask = (regions == marker)
            if not np.any(mask):
                continue
            
            # Get model type from config
            model_type = self.regions_config[marker].get('model_type', 'archie').lower()
            
            # Calculate resistivity based on model type
            if model_type == 'archie':
                resistivity[mask] = model.calculate_resistivity(
                    saturation[mask],
                    fluid_resistivity=kwargs.get('fluid_resistivity', 1.0),
                    porosity=porosity[mask]
                )
            elif model_type == 'waxman_smits':
                resistivity[mask] = model.calculate_resistivity(
                    saturation[mask],
                    base_resistivity=kwargs.get('base_resistivity', 100.0)[mask] if hasattr(kwargs.get('base_resistivity', 100.0), '__len__') else kwargs.get('base_resistivity', 100.0),
                    Qv=kwargs.get('Qv', 0.1)[mask] if hasattr(kwargs.get('Qv', 0.1), '__len__') else kwargs.get('Qv', 0.1)
                )
            elif model_type == 'modified_ws':
                resistivity[mask] = model.calculate_resistivity(
                    saturation[mask],
                    base_resistivity=kwargs.get('base_resistivity', 100.0)[mask] if hasattr(kwargs.get('base_resistivity', 100.0), '__len__') else kwargs.get('base_resistivity', 100.0),
                    surface_conductivity=kwargs.get('surface_conductivity', 0.01)[mask] if hasattr(kwargs.get('surface_conductivity', 0.01), '__len__') else kwargs.get('surface_conductivity', 0.01)
                )
        
        return resistivity
    
    def estimate_saturation(self, resistivity: np.ndarray, porosity: np.ndarray,
                          regions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Estimate saturation from resistivity using region-specific models.
        
        Args:
            resistivity: Formation resistivity values
            porosity: Porosity values
            regions: Region markers for each cell
            **kwargs: Additional parameters for specific models
                
        Returns:
            Estimated water saturation values
        """
        saturation = np.zeros_like(resistivity, dtype=float)
        
        # Estimate saturation for each region
        for marker, model in self.region_models.items():
            # Get mask for this region
            mask = (regions == marker)
            if not np.any(mask):
                continue
            
            # Get model type from config
            model_type = self.regions_config[marker].get('model_type', 'archie').lower()
            
            # Estimate saturation based on model type
            if model_type == 'archie':
                saturation[mask] = model.estimate_saturation(
                    resistivity[mask],
                    fluid_resistivity=kwargs.get('fluid_resistivity', 1.0),
                    porosity=porosity[mask]
                )
            elif model_type == 'waxman_smits':
                saturation[mask] = model.estimate_saturation(
                    resistivity[mask],
                    base_resistivity=kwargs.get('base_resistivity', 100.0)[mask] if hasattr(kwargs.get('base_resistivity', 100.0), '__len__') else kwargs.get('base_resistivity', 100.0),
                    Qv=kwargs.get('Qv', 0.1)[mask] if hasattr(kwargs.get('Qv', 0.1), '__len__') else kwargs.get('Qv', 0.1)
                )
            elif model_type == 'modified_ws':
                saturation[mask] = model.estimate_saturation(
                    resistivity[mask],
                    base_resistivity=kwargs.get('base_resistivity', 100.0)[mask] if hasattr(kwargs.get('base_resistivity', 100.0), '__len__') else kwargs.get('base_resistivity', 100.0),
                    surface_conductivity=kwargs.get('surface_conductivity', 0.01)[mask] if hasattr(kwargs.get('surface_conductivity', 0.01), '__len__') else kwargs.get('surface_conductivity', 0.01)
                )
        
        return saturation


def calculate_resistivity_archie(fluid_conductivity, m=2, porosity=0.2, S_w=1, n=2):
    """
    Calculate the formation resistivity using a modified Archie's Law,
    with fluid conductivity as an input.

    Args:
        fluid_conductivity (float): Conductivity of the formation fluid (S/m).
        S_w (float): Water saturation of the formation. Default is 1 (100% saturated).
        n (float): Saturation exponent. Reflects the effect of water saturation on resistivity. Default is 2.
        m (float): Cementation exponent. Reflects the effect of porosity on resistivity. Default is 2.
        porosity (float): Porosity of the formation. Default is 0.2.

    Returns:
        float: The calculated resistivity of the formation (ohm-m).
    """
    # Convert fluid conductivity to resistivity
    R_w = 1 / fluid_conductivity

    # Calculate the Formation Resistivity Factor (F)
    F = porosity ** (-m)

    # Calculate the resistivity of the formation when fully saturated with water
    R_o = F * R_w

    # Calculate the adjusted resistivity for water saturation
    resistivity = R_o / S_w ** n

    return resistivity


def waxman_smits_resistivity(S_w=1, rho_s=100, n=2, sigma_sur=0):
    """
    Calculate the formation resistivity using the Waxman-Smits model, accounting for the conductivity due to clay content.

    Args:
        S_w (float): Water saturation of the formation. Default is 1 (100% saturated).
        rho_s (float): Resistivity of the formation fully saturated with water (ohm-m). Default is 100 ohm-m.
        n (float): Saturation exponent. Reflects the effect of water saturation on resistivity. Default is 2.
        sigma_sur (float): Surface conductivity due to clay content (mS/m). Default is 0.

    Returns:
        float: The calculated resistivity of the formation (ohm-m), considering both water and clay conductivity.
    """
    # Calculate the conductivity of the formation when fully saturated with water
    sigma_sat = 1 / rho_s  # Convert resistivity to conductivity

    # Adjust the conductivity for water saturation and surface conductivity due to clay
    sigma = sigma_sat * S_w ** n + sigma_sur * S_w ** (n - 1)

    # Calculate the resistivity from the adjusted conductivity
    resistivity = 1 / sigma

    return resistivity


def estimate_saturation_from_resistivity_Ro(rho, R_o, n=2):
    """
    Estimate water saturation from resistivity using a modified Archie's Law,
    with the resistivity at full saturation as an input.

    Args:
        rho (float): Measured resistivity of the formation (ohm-m).
        R_o (float): Resistivity of the formation fully saturated with water (ohm-m).
        n (float): Saturation exponent. Reflects the effect of water saturation on resistivity. Default is 2.


    Returns:
        float: Estimated water saturation of the formation.
    """
    # Inverse Archie's Law to estimate water saturation from resistivity
    S_w = ((rho / R_o) ** (1/n))
    return S_w


def estimate_saturation_fsolve_Ro(rho, R_o, sigma_sur, n=2):
    """
    Estimate water saturation from resistivity in the presence of surface conductivity,
    using the fsolve function from SciPy for numerical solving and R_o as an input.

    Args:
        rho (float or array-like): Measured resistivity of the formation (ohm-m).
        R_o (float): Resistivity of the formation fully saturated with water (ohm-m).
        sigma_sur (float or array-like): Surface conductivity due to clay content (mS/m).
        n (float): Saturation exponent. Default is 2.


    Returns:
        array: Estimated water saturation of the formation for each rho and sigma_sur pair.
    """
    # Ensure inputs are arrays for vectorized operations
    rho = np.asarray(rho)
    sigma_sur = np.asarray(sigma_sur)
    S_t = ((rho / R_o) ** (1/n))
    
    # Define the function to find the root, representing the equation to solve
    def equation_to_solve(S_w, rho, R_o, sigma_sur, n):
        return (R_o * S_w ** (-n)) + (sigma_sur * S_w ** (n - 1)) - 1 / rho

    # Solve for S_w for each rho and sigma_sur
    solution = [fsolve(equation_to_solve, x0=S_t, args=(rho_val, R_o, sigma_sur_val, n))[0] 
                for rho_val, sigma_sur_val in zip(rho, sigma_sur)]

    return np.array(solution)
