"""
Module for processing MODFLOW model outputs.
"""
import os
import numpy as np
from typing import Tuple, Optional, Union, List, Dict

from .base import HydroModelOutput


def binaryread(file, vartype, shape=(1,), charlen=16):
    """
    Uses numpy to read from binary file. This was found to be faster than the
    struct approach and is used as the default.

    Args:
        file: Open file object in binary read mode
        vartype: Variable type to read
        shape: Shape of the data to read (default: (1,))
        charlen: Length of character strings (default: 16)

    Returns:
        The read data
    """
    # Read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1)
    else:
        # Find the number of values
        nval = np.prod(shape)
        result = np.fromfile(file, vartype, nval)
        if nval == 1:
            result = result  # [0]
        else:
            result = np.reshape(result, shape)
    return result


class MODFLOWWaterContent(HydroModelOutput):
    """Class for processing water content data from MODFLOW simulations."""
    
    def __init__(self, model_directory: str, idomain: np.ndarray):
        """
        Initialize MODFLOWWaterContent processor.
        
        Args:
            model_directory: Path to simulation workspace
            idomain: Domain array indicating active cells
        """
        super().__init__(model_directory)
        self.idomain = idomain
        self.nrows, self.ncols = idomain.shape
        
        # Build reverse lookup dictionary (only for first layer as in original code)
        self.iuzno_dict_rev = {}
        iuzno = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                if idomain[i, j] != 0:
                    self.iuzno_dict_rev[iuzno] = (i, j)
                    iuzno += 1
        
        # Store number of UZ flow cells
        self.nuzfcells = len(self.iuzno_dict_rev)
    
    def load_timestep(self, timestep_idx: int, nlay: int = 3) -> np.ndarray:
        """
        Load water content for a specific timestep.
        
        Args:
            timestep_idx: Index of the timestep to load
            nlay: Number of layers in the model
            
        Returns:
            Water content array with shape (nlay, nrows, ncols)
        """
        return self.load_time_range(timestep_idx, timestep_idx + 1, nlay)[0]
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, 
                      nlay: int = 3) -> np.ndarray:
        """
        Load water content for a range of timesteps.
        
        Args:
            start_idx: Starting timestep index (default: 0)
            end_idx: Ending timestep index (exclusive, default: None loads all)
            nlay: Number of layers in the model (default: 3)
            
        Returns:
            Water content array with shape (timesteps, nlay, nrows, ncols)
        """
        # Calculate total UZ flow cells
        nuzfcells = self.nuzfcells * nlay
        
        # Open water content file
        fpth = os.path.join(self.model_directory, "WaterContent")
        file = open(fpth, "rb")
        
        WC_tot = []
        
        # Skip to starting timestep
        for _ in range(start_idx):
            try:
                # Read header
                vartype = [
                    ("kstp", "<i4"),
                    ("kper", "<i4"), 
                    ("pertim", "<f8"),
                    ("totim", "<f8"),
                    ("text", "S16"),
                    ("maxbound", "<i4"),
                    ("1", "<i4"),
                    ("11", "<i4"),
                ]
                binaryread(file, vartype)
                
                # Skip data for this timestep
                vartype = [("data", "<f8")]
                for _ in range(nuzfcells):
                    binaryread(file, vartype)
            except Exception:
                print(f"Error skipping to timestep {start_idx}")
                file.close()
                return np.array(WC_tot)
        
        # Read timesteps
        timestep = 0
        while True:
            # Break if we've read the requested number of timesteps
            if end_idx is not None and timestep >= (end_idx - start_idx):
                break
                
            try:
                # Read header information
                vartype = [
                    ("kstp", "<i4"),
                    ("kper", "<i4"), 
                    ("pertim", "<f8"),
                    ("totim", "<f8"),
                    ("text", "S16"),
                    ("maxbound", "<i4"),
                    ("1", "<i4"),
                    ("11", "<i4"),
                ]
                header = binaryread(file, vartype)
                
                # Initialize water content array for this timestep
                WC_arr = np.zeros((nlay, self.nrows, self.ncols)) * np.nan
                
                # Read water content data
                vartype = [("data", "<f8")]
                
                # Read data for each layer and cell
                for k in range(nlay):
                    for n in range(self.nuzfcells):
                        i, j = self.iuzno_dict_rev[n]
                        WC_arr[k, i, j] = np.array(binaryread(file, vartype).tolist())
                
                WC_tot.append(WC_arr)
                timestep += 1
                
            except Exception as e:
                print(f"Reached end of file or error at timestep {timestep}: {str(e)}")
                break
        
        file.close()
        
        return np.array(WC_tot)
    
    def get_timestep_info(self) -> List[Tuple[int, int, float, float]]:
        """
        Get information about each timestep in the WaterContent file.
        
        Returns:
            List of tuples (kstp, kper, pertim, totim) for each timestep
        """
        # Open water content file
        fpth = os.path.join(self.model_directory, "WaterContent")
        file = open(fpth, "rb")
        
        timestep_info = []
        nuzfcells = self.nuzfcells * 3  # Assuming 3 layers by default
        
        while True:
            try:
                # Read header information
                vartype = [
                    ("kstp", "<i4"),
                    ("kper", "<i4"), 
                    ("pertim", "<f8"),
                    ("totim", "<f8"),
                    ("text", "S16"),
                    ("maxbound", "<i4"),
                    ("1", "<i4"),
                    ("11", "<i4"),
                ]
                header = binaryread(file, vartype)
                
                # Extract timestep info
                kstp = header[0][0]
                kper = header[0][1]
                pertim = header[0][2]
                totim = header[0][3]
                
                timestep_info.append((kstp, kper, pertim, totim))
                
                # Skip data for this timestep
                vartype = [("data", "<f8")]
                for _ in range(nuzfcells):
                    binaryread(file, vartype)
                    
            except Exception:
                break
        
        file.close()
        return timestep_info


class MODFLOWPorosity(HydroModelOutput):
    """Class for processing porosity data from MODFLOW simulations."""
    
    def __init__(self, model_directory: str, model_name: str):
        """
        Initialize MODFLOWPorosity processor.
        
        Args:
            model_directory: Path to simulation workspace
            model_name: Name of the MODFLOW model
        """
        super().__init__(model_directory)
        self.model_name = model_name
        
        try:
            import flopy
            self.flopy_available = True
        except ImportError:
            self.flopy_available = False
            raise ImportError("flopy is required to load MODFLOW porosity data. Please install flopy.")
        
        # Use flopy to load the MODFLOW model
        try:
            import flopy
            self.model = flopy.modflow.Modflow.load(
                f"{model_name}.nam",
                model_ws=model_directory,
                load_only=["UPW", "LPF"],  # Load only packages with porosity
                check=False
            )
            
            # Extract grid information
            self.nlay = self.model.nlay
            self.nrow = self.model.nrow
            self.ncol = self.model.ncol
            
        except Exception as e:
            raise ValueError(f"Error loading MODFLOW model: {str(e)}")
    
    def load_porosity(self) -> np.ndarray:
        """
        Load porosity data from MODFLOW model.
        
        Returns:
            3D array of porosity values (nlay, nrow, ncol)
        """
        try:
            import flopy
            
            # Try to get porosity from UPW package first
            if hasattr(self.model, 'upw') and self.model.upw is not None:
                if hasattr(self.model.upw, 'sy'):
                    return self.model.upw.sy.array
            
            # Then try LPF package
            if hasattr(self.model, 'lpf') and self.model.lpf is not None:
                if hasattr(self.model.lpf, 'sy'):
                    return self.model.lpf.sy.array
            
            # If specific yield not found, try specific storage
            if hasattr(self.model, 'upw') and self.model.upw is not None:
                if hasattr(self.model.upw, 'ss'):
                    print("WARNING: Using specific storage as substitute for porosity")
                    return self.model.upw.ss.array
            
            if hasattr(self.model, 'lpf') and self.model.lpf is not None:
                if hasattr(self.model.lpf, 'ss'):
                    print("WARNING: Using specific storage as substitute for porosity")
                    return self.model.lpf.ss.array
            
            # If nothing found, try default value            
            print("WARNING: No porosity data found in model. Using default value of 0.3")
            return np.ones((self.nlay, self.nrow, self.ncol)) * 0.3
                
        except Exception as e:
            raise ValueError(f"Error loading porosity data: {str(e)}")
    
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load porosity for a specific timestep.
        Note: For MODFLOW, porosity is typically constant over time,
        so this returns the same array regardless of timestep.
        
        Args:
            timestep_idx: Index of the timestep (unused)
            
        Returns:
            3D array of porosity values
        """
        return self.load_porosity()
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Load porosity for a range of timesteps.
        Since porosity is typically constant, this returns a stack of identical arrays.
        
        Args:
            start_idx: Starting timestep index (unused)
            end_idx: Ending timestep index (unused)
            
        Returns:
            4D array of porosity values (nt, nlay, nrow, ncol) where all timesteps are identical
        """
        porosity = self.load_porosity()
        
        # Determine number of timesteps
        if end_idx is None:
            # Load stress period info to determine number of timesteps
            if hasattr(self.model, 'dis') and self.model.dis is not None:
                nt = self.model.dis.nper
            else:
                nt = 1  # Default to single timestep if no info available
        else:
            nt = end_idx - start_idx
        
        # Stack porosity array for each timestep
        return np.tile(porosity[np.newaxis, :, :, :], (nt, 1, 1, 1))
    
    def get_timestep_info(self) -> List[Tuple]:
        """
        Get information about each timestep in the model.
        
        Returns:
            List of tuples (stress_period, timestep, time) for each timestep
        """
        timestep_info = []
        
        if hasattr(self.model, 'dis') and self.model.dis is not None:
            perlen = self.model.dis.perlen.array
            nstp = self.model.dis.nstp.array
            tsmult = self.model.dis.tsmult.array
            
            current_time = 0.0
            for sp in range(len(perlen)):
                if nstp[sp] == 1:
                    # Single timestep in this stress period
                    timestep_info.append((sp, 0, current_time + perlen[sp]))
                    current_time += perlen[sp]
                else:
                    # Multiple timesteps with time step multiplier
                    dt0 = perlen[sp] * (1.0 - tsmult[sp]) / (1.0 - tsmult[sp]**nstp[sp])
                    for stp in range(nstp[sp]):
                        dt = dt0 * tsmult[sp]**stp
                        current_time += dt
                        timestep_info.append((sp, stp, current_time))
        
        return timestep_info