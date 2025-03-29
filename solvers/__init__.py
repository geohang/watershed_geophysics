# Solver modules
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