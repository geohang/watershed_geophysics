"""
Inversion framework for geophysical applications.
"""

# Import inversion base classes
from watershed_geophysics.inversion.base import (
    InversionBase,
    InversionResult,
    TimeLapseInversionResult
)

# Import ERT inversion classes
from watershed_geophysics.inversion.ert_inversion import (
    ERTInversion
)

# Import time-lapse inversion classes
from watershed_geophysics.inversion.time_lapse import (
    TimeLapseERTInversion
)

# Import windowed inversion classes
from watershed_geophysics.inversion.windowed import (
    WindowedTimeLapseERTInversion
)

__all__ = [
    # Base classes
    'InversionBase',
    'InversionResult',
    'TimeLapseInversionResult',
    
    # ERT inversion
    'ERTInversion',
    
    # Time-lapse inversion
    'TimeLapseERTInversion',
    
    # Windowed inversion
    'WindowedTimeLapseERTInversion'
]