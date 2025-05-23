"""
WatershedGeo - A comprehensive package for geophysical modeling and inversion in watershed monitoring.

This package integrates MODFLOW hydrological model outputs with geophysical forward modeling
and inversion, specializing in electrical resistivity tomography (ERT) and seismic velocity modeling.
"""

__version__ = '0.1.0'

# Core utilities
from watershed_geophysics.core.interpolation import (
    ProfileInterpolator, 
    interpolate_to_profile,
    setup_profile_coordinates,
    interpolate_structure_to_profile,
    prepare_2D_profile_data,
    interpolate_to_mesh
)

from watershed_geophysics.core.mesh_utils import (
    MeshCreator,
    create_mesh_from_layers
)

# Import from model_output module
from watershed_geophysics.model_output.modflow_output import MODFLOWWaterContent, MODFLOWPorosity
from watershed_geophysics.model_output.base import HydroModelOutput

# Try to import ParFlow classes if available
try:
    from watershed_geophysics.model_output.parflow_output import ParflowSaturation, ParflowPorosity
except ImportError:
    pass




# Forward modeling
from watershed_geophysics.forward.ert_forward import (
    ERTForwardModeling,
    ertforward,
    ertforward2,
    ertforandjac,
    ertforandjac2
)

# Inversion utilities
from watershed_geophysics.inversion.base import (
    InversionResult,
    TimeLapseInversionResult,
    InversionBase
)

from watershed_geophysics.inversion.ert_inversion import (
    ERTInversion
)

from watershed_geophysics.inversion.time_lapse import (
    TimeLapseERTInversion
)

from watershed_geophysics.inversion.windowed import (
    WindowedTimeLapseERTInversion
)

# Linear solvers
from watershed_geophysics.solvers.linear_solvers import (
    generalized_solver,
    LinearSolver,
    CGLSSolver,
    LSQRSolver,
    RRLSQRSolver,
    RRLSSolver,
    direct_solver,
    TikhonvRegularization,
    IterativeRefinement,
    get_optimal_solver
)

# Petrophysics - Resistivity models
from watershed_geophysics.petrophysics.resistivity_models import (
    water_content_to_resistivity,
    resistivity_to_water_content,
    resistivity_to_saturation,

)

# Petrophysics - Velocity models
from watershed_geophysics.petrophysics.velocity_models import (
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

# Define what gets imported with 'from watershed_geophysics import *'
__all__ = [
    # Core - Interpolation
    'ProfileInterpolator', 
    'interpolate_to_profile',
    'setup_profile_coordinates',
    'interpolate_structure_to_profile',
    'prepare_2D_profile_data',
    'interpolate_to_mesh',
    
    # Core - Mesh
    'MeshCreator',
    'create_mesh_from_layers',
    
    # MODFLOW
    'MODFLOWWaterContent',
    'binaryread',
    
    # Forward modeling
    'ERTForwardModeling',
    'ertforward',
    'ertforward2',
    'ertforandjac',
    'ertforandjac2',
    
    # Inversion base
    'InversionResult',
    'TimeLapseInversionResult',
    'InversionBase',
    
    # ERT inversion
    'ERTInversion',
    
    # Time-lapse inversion
    'TimeLapseERTInversion',
    
    # Windowed inversion
    'WindowedTimeLapseERTInversion',
    
    # Linear solvers
    'generalized_solver',
    'LinearSolver',
    'CGLSSolver',
    'LSQRSolver',
    'RRLSQRSolver',
    'RRLSSolver',
    'direct_solver',
    'TikhonvRegularization',
    'IterativeRefinement',
    'get_optimal_solver',
    
    # Resistivity models
    'BaseResistivityModel',
    'ArchieModel',
    'WaxmanSmitsModel',
    'ModifiedWaxmanSmits',
    'HybridResistivityModel',
    'calculate_resistivity_archie',
    'waxman_smits_resistivity',
    'estimate_saturation_from_resistivity_Ro',
    'estimate_saturation_fsolve_Ro',
    
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