"""
Petrophysics module for converting between physical properties of rocks and geophysical observables.
"""

# Import from resistivity_models
from .resistivity_models import (
    water_content_to_resistivity,
    resistivity_to_water_content,
    resistivity_to_saturation
)

# Import from velocity_models
from .velocity_models import (
    BaseVelocityModel,
    VRHModel,
    BrieModel,
    DEMModel,
    HertzMindlinModel,
    VRH_model,
    satK,
    velDEM,
    vel_porous
)

__all__ = [
    # Resistivity models
    'water_content_to_resistivity',
    'resistivity_to_water_content',
    'resistivity_to_saturation',

    # Velocity models
    'BaseVelocityModel',
    'VRHModel',
    'BrieModel',
    'DEMModel',
    'HertzMindlinModel',
    'VRH_model',
    'satK',
    'velDEM',
    'vel_porous'
]
