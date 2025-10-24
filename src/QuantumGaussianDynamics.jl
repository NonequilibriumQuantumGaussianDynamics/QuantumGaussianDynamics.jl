"""
QuantumGaussianDynamics.jl 

This package integrates the Time-Dependent Self-Consistent Harmonic Approximation (TDSCHA)
equations of motion to simulate pump–probe-like dynamics. 

Main components:
- `Dynamics` — simulation settings (time step, total time, algorithm, I/O).
- `ElectricField` — external IR electric field coupling (effective charges, direction, time profile).
- `WignerDistribution` — quantum state of the system.
- `Ensemble` — stochastic configurations, forces, stresses, and weights.

**Units**
We use Rydberg atomic units internally (frequencies in Ry units, lengths in Bohr).
Helper constants (e.g. `CONV_BOHR`, `CONV_FS`, `CONV_RY`) are provided to convert common lab units.

See the docstrings of each type/function for details and examples.
"""

module QuantumGaussianDynamics

using LinearAlgebra
using Random
using Roots
using MPI

using Optim
using OptimizationOptimJL
using ForwardDiff

using PyCall

# Init MPI
include("parallel.jl")
include("constants.jl")
include("dynamics.jl")
include("wigner.jl")
include("time_evolution.jl")
include("ensemble.jl")
include("phonons.jl")
include("calculator.jl")
include("external_f.jl")

# Core structures
export Dynamics, Ensemble, ElectricField, WignerDistribution

# Initialization functions
export init_from_dyn, init_ensemble_from_python, init_calculator, fake_field

# Core workflow
export generate_ensemble!, calculate_ensemble!, get_averages!, get_alphabeta, get_correlators
export integrate!

end 
