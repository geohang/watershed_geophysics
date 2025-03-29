"""
Interpolation utilities for geophysical data processing.
"""
import numpy as np
from scipy.interpolate import griddata
from typing import Tuple, List, Optional, Union


def interpolate_to_profile(data: np.ndarray, 
                         X_grid: np.ndarray, 
                         Y_grid: np.ndarray,
                         X_pro: np.ndarray, 
                         Y_pro: np.ndarray,
                         method: str = 'linear') -> np.ndarray:
    """
    Interpolate 2D data onto a profile line
    
    Args:
        data: 2D array of values to interpolate
        X_grid: X coordinates of original grid (meshgrid)
        Y_grid: Y coordinates of original grid (meshgrid)
        X_pro: X coordinates of profile points
        Y_pro: Y coordinates of profile points
        method: Interpolation method ('linear' or 'nearest')
        
    Returns:
        Interpolated values along profile
    """
    X_new = X_grid.ravel()
    Y_new = Y_grid.ravel()
    
    return griddata((X_new, Y_new), data.ravel(),
                   (X_pro.ravel(), Y_pro.ravel()),
                   method=method)


def setup_profile_coordinates(point1: List[int], 
                            point2: List[int],
                            surface_data: np.ndarray,
                            origin_x: float = 0.0,
                            origin_y: float = 0.0,
                            pixel_width: float = 1.0,
                            pixel_height: float = -1.0,
                            num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up profile coordinates based on surface elevation data between two points
    
    Args:
        point1: Starting point indices [col, row]
        point2: Ending point indices [col, row]
        surface_data: 2D array of surface elevation data
        origin_x: X coordinate of origin
        origin_y: Y coordinate of origin
        pixel_width: Width of each pixel
        pixel_height: Height of each pixel (negative for top-down)
        num_points: Number of points along profile
        
    Returns:
        X_pro: X coordinates along profile
        Y_pro: Y coordinates along profile
        L_profile: Distances along profile
        XX: X coordinate grid
        YY: Y coordinate grid
    """
    # Create coordinate grids
    x = origin_x + pixel_width * np.arange(surface_data.shape[1])
    y = origin_y + pixel_height * np.arange(surface_data.shape[0])
    XX, YY = np.meshgrid(x, y)
    
    # Handle no-data values
    surface_data = surface_data.copy()
    surface_data[surface_data == 0] = np.nan
    
    # Calculate start and end positions
    P1_pos = np.array([x[point1[0]], y[point1[1]]])
    P2_pos = np.array([x[point2[0]], y[point2[1]]])
    
    # Calculate total distance
    dis = np.sqrt(np.sum((P1_pos - P2_pos)**2))
    
    # Generate profile coordinates
    X_pro = (x[point1[0]] - x[point2[0]])/dis * np.linspace(0, dis, num_points)[:-1] + x[point2[0]]
    Y_pro = (y[point1[1]] - y[point2[1]])/dis * np.linspace(0, dis, num_points)[:-1] + y[point2[1]]
    
    # Calculate profile distances
    L_profile = np.sqrt((X_pro - X_pro[0])**2 + (Y_pro - Y_pro[0])**2)
    
    return X_pro, Y_pro, L_profile, XX, YY


def interpolate_structure_to_profile(structure_data: List[np.ndarray],
                                   X_grid: np.ndarray,
                                   Y_grid: np.ndarray,
                                   X_pro: np.ndarray,
                                   Y_pro: np.ndarray) -> np.ndarray:
    """
    Interpolate multiple structure layers onto profile
    
    Args:
        structure_data: List of 2D arrays for each layer
        X_grid: X coordinates of original grid
        Y_grid: Y coordinates of original grid
        X_pro: X coordinates of profile points
        Y_pro: Y coordinates of profile points
        
    Returns:
        Array of interpolated values with shape (n_layers, n_points)
    """
    structure = []
    for layer in structure_data:
        interpolated = interpolate_to_profile(layer, X_grid, Y_grid,
                                           X_pro, Y_pro)
        structure.append(interpolated)
    return np.array(structure)


def prepare_2D_profile_data(data: np.ndarray, 
                          XX: np.ndarray, 
                          YY: np.ndarray,
                          X_pro: np.ndarray,
                          Y_pro: np.ndarray) -> np.ndarray:
    """
    Interpolate multiple 2D gridded data layers onto a profile line.
    
    Args:
        data: 3D array of gridded data (n_layers, ny, nx)
        XX, YY: Coordinate grids from meshgrid
        X_pro, Y_pro: Profile line coordinates
        
    Returns:
        Interpolated values along profile (n_layers, n_profile_points)
    """
    n_layers = data.shape[0]
    profile_values = []
    
    X_new = XX.ravel()
    Y_new = YY.ravel()
    
    for i in range(n_layers):
        layer_values = griddata((X_new, Y_new), 
                              data[i].ravel(), 
                              (X_pro.ravel(), Y_pro.ravel()), 
                              method='linear')
        profile_values.append(layer_values)
    
    return np.array(profile_values)


def interpolate_to_mesh(property_values: np.ndarray,
                       profile_distance: np.ndarray,
                       depth_values: np.ndarray,
                       mesh_x: np.ndarray,
                       mesh_y: np.ndarray,
                       mesh_markers: np.ndarray,
                       layer_markers: list = [3, 0, 2]) -> np.ndarray:
    """
    Interpolate property values from profile to mesh with layer-specific handling.
    
    Args:
        property_values: Property values array (n_points)
        profile_distance: Distance along profile (n_points)
        depth_values: Depth values array (n_layers, n_points)
        mesh_x: X coordinates of mesh cells
        mesh_y: Y coordinates of mesh cells
        mesh_markers: Markers indicating different layers in mesh
        layer_markers: List of marker values for each layer
    
    Returns:
        Interpolated values for mesh cells
    """
    # Initialize output array
    result = np.zeros_like(mesh_markers, dtype=float)
    
    # Get number of points and layers
    n_points = profile_distance.shape[0]
    n_layers = len(layer_markers)
    
    # Create layer boundaries for interpolation
    layer_boundaries = np.zeros((n_layers+1, n_points))
    for i in range(n_layers):
        layer_boundaries[i] = depth_values[i]
    layer_boundaries[-1] = depth_values[-1]  # Bottom boundary
    
    # Create coordinates and values for each layer
    for i, marker in enumerate(layer_markers):
        # Extract cells for this layer
        layer_mask = mesh_markers == marker
        if not np.any(layer_mask):
            continue
            
        # Get mesh points for this layer
        points_x = mesh_x[layer_mask]
        points_y = mesh_y[layer_mask]
        
        # Create interpolation points
        x_points = np.repeat(profile_distance, 2)
        y_points = np.column_stack((layer_boundaries[i], layer_boundaries[i+1])).flatten()
        
        # Create values array
        if property_values.ndim == 1:
            values = np.repeat(property_values, 2)
        else:
            values = np.repeat(property_values[i], 2)
        
        # Do interpolation
        values_linear = griddata(
            (x_points, y_points),
            values,
            (points_x, points_y),
            method='linear'
        )
        
        # Fill NaN values with nearest neighbor interpolation
        nan_mask = np.isnan(values_linear)
        if np.any(nan_mask):
            values_nearest = griddata(
                (x_points, y_points),
                values,
                (points_x[nan_mask], points_y[nan_mask]),
                method='nearest'
            )
            values_linear[nan_mask] = values_nearest
        
        # Store results
        result[layer_mask] = values_linear
    
    return result


class ProfileInterpolator:
    """Class for handling interpolation of data to/from profiles."""
    
    def __init__(self, point1: List[int], point2: List[int], 
                surface_data: np.ndarray,
                origin_x: float = 0.0, origin_y: float = 0.0,
                pixel_width: float = 1.0, pixel_height: float = -1.0,
                num_points: int = 200):
        """
        Initialize profile interpolator with reference points and surface data.
        
        Args:
            point1: Starting point indices [col, row]
            point2: Ending point indices [col, row]
            surface_data: 2D array of surface elevation data
            origin_x, origin_y: Coordinates of origin
            pixel_width, pixel_height: Pixel dimensions
            num_points: Number of points along profile
        """
        self.point1 = point1
        self.point2 = point2
        self.surface_data = surface_data
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.num_points = num_points
        
        # Set up profile coordinates
        self.X_pro, self.Y_pro, self.L_profile, self.XX, self.YY = setup_profile_coordinates(
            point1, point2, surface_data, origin_x, origin_y, 
            pixel_width, pixel_height, num_points
        )
        
        # Get surface profile
        self.surface_profile = interpolate_to_profile(
            surface_data, self.XX, self.YY, self.X_pro, self.Y_pro
        )
    
    def interpolate_layer_data(self, layer_data: List[np.ndarray]) -> np.ndarray:
        """
        Interpolate multiple layer data to profile.
        
        Args:
            layer_data: List of 2D arrays for each layer
            
        Returns:
            Array of interpolated values (n_layers, n_profile_points)
        """
        return interpolate_structure_to_profile(
            layer_data, self.XX, self.YY, self.X_pro, self.Y_pro
        )
    
    def interpolate_3d_data(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolate 3D data (n_layers, ny, nx) to profile.
        
        Args:
            data: 3D array of values
            
        Returns:
            Array of interpolated values (n_layers, n_profile_points)
        """
        return prepare_2D_profile_data(
            data, self.XX, self.YY, self.X_pro, self.Y_pro
        )
    
    def interpolate_to_mesh(self, property_values: np.ndarray,
                          depth_values: np.ndarray,
                          mesh_x: np.ndarray,
                          mesh_y: np.ndarray,
                          mesh_markers: np.ndarray,
                          layer_markers: list = [3, 0, 2]) -> np.ndarray:
        """
        Interpolate property values from profile to mesh with layer-specific handling.
        
        Args:
            property_values: Property values array (n_points or n_layers, n_points)
            depth_values: Depth values array (n_layers, n_points)
            mesh_x, mesh_y: Coordinates of mesh cells
            mesh_markers: Markers indicating different layers in mesh
            layer_markers: List of marker values for each layer
        
        Returns:
            Interpolated values for mesh cells
        """
        return interpolate_to_mesh(
            property_values, self.L_profile, depth_values,
            mesh_x, mesh_y, mesh_markers, layer_markers
        )
