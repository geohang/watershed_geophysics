#Initial setup file for running the Parflow model.
#
set tcl_precision 17
#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*
pfset FileVersion 4
set runname InitialRunFile
pfset Process.Topology.P        [lindex $argv 0]
pfset Process.Topology.Q        [lindex $argv 1]
pfset Process.Topology.R        [lindex $argv 2]
set dx 5.041667
set dy $dx
set dz 200.705000
set nx 48
set ny 50
set nz 1
set x0 569131.090000
set y0 4842212.330000
set z0 0.000000
set xmax [expr $x0 + ($nx * $dx)]
set ymax [expr $y0 + ($ny * $dy)]
set zmax [expr $z0 + ($nz * $dz)]
#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X                 $x0
pfset ComputationalGrid.Lower.Y                 $y0
pfset ComputationalGrid.Lower.Z                 $z0
pfset ComputationalGrid.DX	                 $dx
pfset ComputationalGrid.DY                      $dy
pfset ComputationalGrid.DZ	                 $dz
pfset ComputationalGrid.NX                      $nx
pfset ComputationalGrid.NY                      $ny
pfset ComputationalGrid.NZ                      $nz
#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names                 "domainbox solidfile"
pfset GeomInput.domainbox.InputType    Box
pfset GeomInput.domainbox.GeomName     background
pfset GeomInput.solidfile.InputType    SolidFile
pfset GeomInput.solidfile.GeomNames    domain
pfset GeomInput.solidfile.FileName     test1.pfsol
pfset Geom.domain.Patches              "Bottom  Top  Usr_2"
#---------------------------------------------------------
# Domain 
#---------------------------------------------------------
pfset Domain.GeomName domain
#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------
pfset Geom.background.Lower.X                        $x0
pfset Geom.background.Lower.Y                        $y0
pfset Geom.background.Lower.Z                        $z0
pfset Geom.background.Upper.X                        $xmax
pfset Geom.background.Upper.Y                        $ymax
pfset Geom.background.Upper.Z                        $zmax
pfset Geom.background.Patches             "x-lower x-upper y-lower y-upper z-lower z-upper"
#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names                 "domain"
pfset Geom.domain.Perm.Type            Constant
pfset Geom.domain.Perm.Value           0.001
pfset Perm.TensorType               TensorByGeom
pfset Geom.Perm.TensorByGeom.Names  "domain"
pfset Geom.domain.Perm.TensorValX  1.0
pfset Geom.domain.Perm.TensorValY  1.0
pfset Geom.domain.Perm.TensorValZ  1.0
#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "domain "
pfset Geom.domain.SpecificStorage.Value 1.0e-6
#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
pfset Phase.Names "water"
pfset Phase.water.Density.Type	Constant
pfset Phase.water.Density.Value	1.0
pfset Phase.water.Viscosity.Type	Constant
pfset Phase.water.Viscosity.Value	1.0
#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names			""
#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfset Geom.Retardation.GeomNames           ""
#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
pfset Gravity				1.0
#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------
pfset Geom.Porosity.GeomNames           "domain"
pfset Geom.domain.Porosity.Type         Constant
pfset Geom.domain.Porosity.Value        0.25
#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
pfset Phase.RelPerm.Type           VanGenuchten
pfset Phase.RelPerm.GeomNames      "domain"
pfset Geom.domain.RelPerm.Alpha    3.47
pfset Geom.domain.RelPerm.N        7.24
#-----------------------------------------------------------------------------
# Saturation
#-----------------------------------------------------------------------------
pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         "domain"
pfset Geom.domain.Saturation.Alpha        3.47
pfset Geom.domain.Saturation.N            7.24
pfset Geom.domain.Saturation.SRes         0.12
pfset Geom.domain.Saturation.SSat         1.0
#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
pfset Phase.water.Mobility.Type        Constant
pfset Phase.water.Mobility.Value       1.0
#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                           ""
#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
# The UNITS on this simulation are HOURS
pfset TimingInfo.BaseUnit        0.01
pfset TimingInfo.StartCount      0
pfset TimingInfo.StartTime       0.0
pfset TimingInfo.StopTime        2.0
pfset TimingInfo.DumpInterval    0.1
pfset TimeStep.Type              Constant
pfset TimeStep.Value             0.01
#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names                       "constant rainrec"
pfset Cycle.constant.Names              "alltime"
pfset Cycle.constant.alltime.Length      200
pfset Cycle.constant.Repeat             -1
# rainfall and recession time periods are defined here
pfset Cycle.rainrec.Names                  "rain rec"
# nrain/BaseUnit
pfset Cycle.rainrec.rain.Length            100
# nrec/BaseUnit
pfset Cycle.rainrec.rec.Length             100
pfset Cycle.rainrec.Repeat                 -1
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames              "Bottom  Top  Usr_2"
pfset Patch.Bottom.BCPressure.Type		     FluxConst
pfset Patch.Bottom.BCPressure.Cycle		      "constant"
pfset Patch.Bottom.BCPressure.alltime.Value	      0.0
pfset Patch.Top.BCPressure.Type             OverlandFlow
pfset Patch.Top.BCPressure.Cycle	      "rainrec"
pfset Patch.Top.BCPressure.rain.Value	      -0.005
pfset Patch.Top.BCPressure.rec.Value	      0.000
pfset Patch.Usr_2.BCPressure.Type		     FluxConst
pfset Patch.Usr_2.BCPressure.Cycle		      "constant"
pfset Patch.Usr_2.BCPressure.alltime.Value	      0.0
#-----------------------------------------------------------------------------
# Topo slopes in x-direction
#-----------------------------------------------------------------------------
file copy -force "slx.pfb" slope_x.pfb
pfset TopoSlopesX.Type                 "PFBFile"
pfset TopoSlopesX.GeomNames            "domain"
pfset TopoSlopesX.FileName              slope_x.pfb
pfdist -nz 1 slope_x.pfb
#-----------------------------------------------------------------------------
# Topo slopes in y-direction
#-----------------------------------------------------------------------------
file copy -force "sly.pfb" slope_y.pfb
pfset TopoSlopesY.Type                 "PFBFile"
pfset TopoSlopesY.GeomNames            "domain"
pfset TopoSlopesY.FileName              slope_y.pfb
pfdist -nz 1 slope_y.pfb
#-----------------------------------------------------------------------------
# Mannings coefficient 
#-----------------------------------------------------------------------------
set mn 0.000033
pfset Mannings.Type "Constant"
pfset Mannings.GeomNames "domain"
pfset Mannings.Geom.domain.Value $mn
#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    domain
pfset PhaseSources.water.Geom.domain.Value            0.0
#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
pfset KnownSolution                                    NoKnownSolution
#-----------------------------------------------------------------------------
# Initial conditions: water pressure
#-----------------------------------------------------------------------------
pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      -1.0
pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   Top
#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver                                             Richards
pfset Solver.MaxIter                                     2500000
pfset OverlandFlowDiffusive                              0
pfset Solver.Nonlinear.MaxIter                           1000
pfset Solver.Nonlinear.ResidualTol                       1e-10
pfset Solver.Nonlinear.EtaChoice                         Walker1
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.001
pfset Solver.Nonlinear.UseJacobian                       False
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-16
pfset Solver.Nonlinear.StepTol		             1e-30
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      50
pfset Solver.Linear.MaxRestart                           3
pfset Solver.Linear.Preconditioner                       PFMG
pfset Solver.Linear.Preconditioner.PFMG.MaxIter           5
pfset Solver.Linear.Preconditioner.PFMG.Smoother          RBGaussSeidelNonSymmetric
pfset Solver.Linear.Preconditioner.PFMG.NumPreRelax       1
pfset Solver.Linear.Preconditioner.PFMG.NumPostRelax      1
pfset Solver.Drop                                       1E-20
pfset Solver.AbsTol                                      1E-9
pfset Solver.WriteSiloSubsurfData True
pfset Solver.WriteSiloPressure True
pfset Solver.WriteSiloSaturation True
pfset Solver.WriteSiloConcentration True
#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfrun $runname
pfundist $runname
