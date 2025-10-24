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

const SMALL_VALUE::Float64 = 1e-8
const THR_ACOUSTIC::Float64 = 1e-1
const CONV_FS::Float64 = 0.048377687 # to femtoseconds
const CONV_RY::Float64 = 0.0734985857 # eV to Ry
const CONV_BOHR::Float64 = 1.8897261246257702 # Angstrom to Bohr
const CONV_MASS::Float64 = 911.444175 # amu to kg
const CONV_EFIELD::Float64 = 2.7502067*1e-7 #kVcm to E_Ry
const CONV_FREQ::Float64 = 4.83776857*1e-5 #THz to w_Ry

export SMALL_VALUE
export CONV_FS
export CONV_RY
export CONV_BOHR
export CONV_EFIELD

include("wigner.jl")
include("time_evolution.jl")
include("ensemble.jl")
include("phonons.jl")
include("calculator.jl")
include("dynamics.jl")
include("external_f.jl")

# Core structures
export Dynamics, Ensemble, ElectricField, WignerDistribution

# Initialization functions
export init_from_dyn, init_ensemble_from_python, init_calculator, fake_field

# Core workflow
export generate_ensemble!, calculate_ensemble!, get_averages!, get_alphabeta, get_correlators
export integrate!

end 
