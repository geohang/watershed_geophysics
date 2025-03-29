"""
Forward modeling utilities for Electrical Resistivity Tomography (ERT).
"""
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from typing import Tuple, Optional, Union


class ERTForwardModeling:
    """Class for forward modeling of Electrical Resistivity Tomography (ERT) data."""
    
    def __init__(self, mesh: pg.Mesh, data: Optional[pg.DataContainer] = None):
        """
        Initialize ERT forward modeling.
        
        Args:
            mesh: PyGIMLI mesh for forward modeling
            data: ERT data container
        """
        self.mesh = mesh
        self.data = data
        self.fwd_operator = ert.ERTModelling()
        
        if data is not None:
            self.fwd_operator.setData(data)
        
        self.fwd_operator.setMesh(mesh)
    
    def set_data(self, data: pg.DataContainer) -> None:
        """
        Set ERT data for forward modeling.
        
        Args:
            data: ERT data container
        """
        self.data = data
        self.fwd_operator.setData(data)
    
    def set_mesh(self, mesh: pg.Mesh) -> None:
        """
        Set mesh for forward modeling.
        
        Args:
            mesh: PyGIMLI mesh
        """
        self.mesh = mesh
        self.fwd_operator.setMesh(mesh)
    
    def forward(self, resistivity_model: np.ndarray, log_transform: bool = True) -> np.ndarray:
        """
        Compute forward response for a given resistivity model.
        
        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Forward response (apparent resistivity)
        """
        # Convert to PyGIMLI RVector if needed
        if isinstance(resistivity_model, np.ndarray):
            model = pg.Vector(resistivity_model.ravel())
        else:
            model = resistivity_model
            
        # Apply exponentiation if log-transformed input
        if log_transform:
            model = pg.Vector(np.exp(model))
            
        # Calculate response
        response = self.fwd_operator.response(model)
        
        # Log-transform response if requested
        if log_transform:
            return np.log(response.array())
        
        return response.array()
    
    def forward_and_jacobian(self, resistivity_model: np.ndarray, log_transform: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward response and Jacobian matrix.
        
        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Tuple of (forward response, Jacobian matrix)
        """
        # Convert to PyGIMLI RVector if needed
        if isinstance(resistivity_model, np.ndarray):
            model = pg.Vector(resistivity_model.ravel())
        else:
            model = resistivity_model
            
        # Apply exponentiation if log-transformed input
        if log_transform:
            model = pg.Vector(np.exp(model))
        
        # Calculate response
        response = self.fwd_operator.response(model)
        
        # Create Jacobian matrix
        self.fwd_operator.createJacobian(model)
        jacobian = self.fwd_operator.jacobian()
        J = pg.utils.gmat2numpy(jacobian)
        
        # Process Jacobian according to log transformations
        if log_transform:
            # For log-transformed model and response
            # J_log = J * exp(m) / d = d(log(d))/d(log(m))
            J = np.exp(resistivity_model.ravel()) * J
            response_array = response.array()
            J = J / response_array.reshape(response_array.shape[0], 1)
            
            return np.log(response.array()), J
        
        return response.array(), J
    
    def get_coverage(self, resistivity_model: np.ndarray, log_transform: bool = True) -> np.ndarray:
        """
        Compute coverage (resolution) for a given resistivity model.
        
        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Coverage values for each cell
        """
        # Convert to PyGIMLI RVector if needed
        if isinstance(resistivity_model, np.ndarray):
            model = pg.Vector(resistivity_model.ravel())
        else:
            model = resistivity_model
            
        # Apply exponentiation if log-transformed input
        if log_transform:
            model = pg.Vector(np.exp(model))
        
        # Calculate response and Jacobian
        response = self.fwd_operator.response(model)
        self.fwd_operator.createJacobian(model)
        
        # Calculate coverage
        covTrans = pg.core.coverageDCtrans(
            self.fwd_operator.jacobian(), 
            1.0 / response, 
            1.0 / model
        )
        
        # Weight by cell sizes
        paramSizes = np.zeros(len(model))
        mesh = self.fwd_operator.paraDomain
        
        for c in mesh.cells():
            paramSizes[c.marker()] += c.size()
            
        FinalJ = np.log10(covTrans / paramSizes)
        
        return FinalJ


def ertforward(fob, mesh, rhomodel, xr):
    """
    Forward model for ERT.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        mesh (pg.Mesh): Mesh for the forward model.
        rhomodel (pg.RVector): Resistivity model vector.
        xr (np.ndarray): Log-transformed model parameter (resistivity).

    Returns:
        dr (np.ndarray): Log-transformed forward response.
        rhomodel (pg.RVector): Updated resistivity model.
    """
    xr1 = np.log(rhomodel.array())
    xr1[mesh.cellMarkers() == 2] = np.exp(xr)
    rhomodel = pg.matrix.RVector(xr1)
    dr = fob.response(rhomodel)
    return np.log(dr.array()), rhomodel


def ertforward2(fob, xr, mesh):
    """
    Simplified ERT forward model.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.

    Returns:
        dr (np.ndarray): Log-transformed forward response.
    """
    xr1 = xr.copy()
    xr1 = np.exp(xr)
    rhomodel = xr1

    dr = fob.response(rhomodel)
    dr = np.log(dr)
    return dr


def ertforandjac(fob, rhomodel, xr):
    """
    Forward model and Jacobian for ERT.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        rhomodel (pg.RVector): Resistivity model.
        xr (np.ndarray): Log-transformed model parameter.

    Returns:
        dr (np.ndarray): Log-transformed forward response.
        J (np.ndarray): Jacobian matrix.
    """
    dr = fob.response(rhomodel)
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = np.exp(xr)*J
    dr = dr.array()
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J


def ertforandjac2(fob, xr, mesh):
    """
    Alternative ERT forward model and Jacobian using log-resistivity values.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.

    Returns:
        dr (np.ndarray): Log-transformed forward response.
        J (np.ndarray): Jacobian matrix.
    """
    xr1 = xr.copy()
    xr1 = np.exp(xr)
    rhomodel = xr1
    dr = fob.response(rhomodel)
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = np.exp(xr.T)*J
    dr = dr.array()
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J
