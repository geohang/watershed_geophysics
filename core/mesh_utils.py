"""
Mesh utilities for geophysical modeling and inversion.
"""
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from typing import Tuple, List, Optional, Union


def create_mesh_from_layers(surface: np.ndarray,
                          line1: np.ndarray,
                          line2: np.ndarray,
                          bottom_depth: float = 30.0,
                          quality: float = 28,
                          area: float = 40) -> Tuple[pg.Mesh, np.ndarray, np.ndarray]:
    """
    Create mesh from layer boundaries and get cell centers and markers.
    
    Args:
        surface: Surface coordinates [[x,z],...] 
        line1: First layer boundary coordinates 
        line2: Second layer boundary coordinates 
        bottom_depth: Depth below surface minimum for mesh bottom
        quality: Mesh quality parameter
        area: Maximum cell area
        
    Returns:
        mesh: PyGIMLI mesh
        mesh_centers: Array of cell center coordinates
        markers: Array of cell markers
    """
    # Calculate bottom elevation from normalized surface
    min_surface_elev = np.nanmin(surface[:,1])
    bottom_elev = bottom_depth #min_surface_elev - bottom_depth
    
    # Create reversed lines for polygon creation
    line1r = line1.copy()
    line1r[:,0] = np.flip(line1[:,0])
    line1r[:,1] = np.flip(line1[:,1])
    
    line2r = line2.copy()
    line2r[:,0] = np.flip(line2[:,0])
    line2r[:,1] = np.flip(line2[:,1])
    
    # Create surface layer
    layer1 = mt.createPolygon(surface,
                             isClosed=False, 
                             marker=2, 
                             boundaryMarker=-1,
                             interpolate='linear', 
                             area=0.1)
    
    # Create middle layer
    Gline1 = mt.createPolygon(np.vstack((line1, line2r)),
                             isClosed=True, 
                             marker=3, 
                             boundaryMarker=1,
                             interpolate='linear', 
                             area=1)
    
    # Create bottom boundary
    Gline2 = mt.createPolygon([[surface[0,0], surface[0,1]],
                              [line2[0,0], bottom_elev],
                              [line2[-1,0], bottom_elev],
                              [surface[-1,0], surface[-1,1]]],
                             isClosed=False, 
                             marker=2, 
                             boundaryMarker=1,
                             interpolate='linear', 
                             area=2)
    
    # Create bottom layer
    layer2 = mt.createPolygon(np.vstack((line2r,
                                        [[line2[0,0], line2[0,1]],
                                         [line2[0,0], bottom_elev],
                                         [line2[-1,0], bottom_elev],
                                         [line2[-1,0], line2[-1,1]]])),
                             isClosed=True, 
                             marker=2, 
                             area=2, 
                             boundaryMarker=1)
    
    # Combine all geometries
    geom = layer1 + layer2 + Gline1 + Gline2
    
    # Create mesh
    mesh = mt.createMesh(geom, quality=quality, area=area)
    
    # Get cell centers and markers
    mesh_centers = np.array(mesh.cellCenters())
    markers = np.array(mesh.cellMarkers())
    
    return mesh, mesh_centers, markers,geom


class MeshCreator:
    """Class for creating and managing meshes for geophysical inversion."""
    
    def __init__(self, quality: float = 28, area: float = 40):
        """
        Initialize MeshCreator with quality and area parameters.
        
        Args:
            quality: Mesh quality parameter (higher is better)
            area: Maximum cell area
        """
        self.quality = quality
        self.area = area
    
    def create_from_layers(self, surface: np.ndarray, 
                          layers: List[np.ndarray],
                          bottom_depth: float = 30.0,
                          markers: List[int] = None) -> pg.Mesh:
        """
        Create a mesh from surface and layer boundaries.
        
        Args:
            surface: Surface coordinates [[x,z],...]
            layers: List of layer boundary coordinates
            bottom_depth: Depth below surface minimum for mesh bottom
            markers: List of markers for each layer (default: [2, 3, 2, ...])
            
        Returns:
            PyGIMLI mesh
        """
        if len(layers) < 1:
            raise ValueError("At least one layer boundary is required")
            
        # Create default markers if not provided
        if markers is None:
            markers = [2] * (len(layers) + 1)
            if len(layers) > 0:
                markers[1] = 3  # Middle layer
        
        # Normalize elevation by maximum elevation
        max_ele = np.nanmax(surface[:,1])
        surface_norm = surface.copy()
        surface_norm[:,1] = surface_norm[:,1]  #- max_ele
        
        layers_norm = []
        for layer in layers:
            layer_norm = layer.copy()
            layer_norm[:,1] = layer_norm[:,1] # - max_ele
            layers_norm.append(layer_norm)
        
        # Create mesh using specific implementation
        if len(layers) == 2:
            mesh, centers, markers_array,geom = create_mesh_from_layers(
                surface_norm, layers_norm[0], layers_norm[1], 
                bottom_depth, self.quality, self.area
            )
            return mesh,geom
        else:
            # Implement custom mesh creation for different number of layers
            raise NotImplementedError("Currently only 2-layer mesh creation is implemented")
    
    def create_from_ert_data(self, data, max_depth: float = 30.0, quality: float = 34):
        """
        Create a mesh suitable for ERT inversion from ERT data.
        
        Args:
            data: PyGIMLI ERT data object
            max_depth: Maximum depth of the mesh
            quality: Mesh quality parameter
            
        Returns:
            PyGIMLI mesh for ERT inversion
        """
        from pygimli.physics import ert
        ert_manager = ert.ERTManager(data)
        return ert_manager.createMesh(data=data, quality=quality)
