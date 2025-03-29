"""
Core utilities for geophysical modeling and inversion.
"""

# Import mesh utilities
from watershed_geophysics.core.mesh_utils import (
    MeshCreator,
    create_mesh_from_layers
)

# Import interpolation utilities
from watershed_geophysics.core.interpolation import (
    ProfileInterpolator,
    interpolate_to_profile,
    setup_profile_coordinates,
    interpolate_structure_to_profile,
    prepare_2D_profile_data,
    interpolate_to_mesh
)

__all__ = [
    # Mesh utilities
    'MeshCreator',
    'create_mesh_from_layers',
    
    # Interpolation utilities
    'ProfileInterpolator',
    'interpolate_to_profile',
    'setup_profile_coordinates',
    'interpolate_structure_to_profile',
    'prepare_2D_profile_data',
    'interpolate_to_mesh'
]