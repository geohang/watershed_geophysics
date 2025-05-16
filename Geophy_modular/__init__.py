"""
Geophysical data processing module for watershed monitoring.
"""

from watershed_geophysics.Geophy_modular.seismic_processor import (
    extract_velocity_structure,
    process_seismic_tomography,
    seismic_velocity_classifier
)

from watershed_geophysics.Geophy_modular.structure_integration import (
    create_ert_mesh_with_structure,
    integrate_velocity_interface,
    create_joint_inversion_mesh
)

from watershed_geophysics.Geophy_modular.ERT_to_WC import (
    ERTtoWC,
    plot_time_series
)


__all__ = [
    # Seismic processing functions
    'extract_velocity_structure',
    'process_seismic_tomography',
    'seismic_velocity_classifier',
    
    # Structure integration functions
    'create_ert_mesh_with_structure',
    'integrate_velocity_interface',
    'create_joint_inversion_mesh'


     # ERT to water content conversion
    'ERTtoWC',
    'plot_time_series'
]