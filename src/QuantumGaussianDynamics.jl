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

# Parallelization module
include("parallel.jl")

# Structures module
include("structures.jl")

# Setting and dynamics loop
include("dynamics.jl")

# Wigner distribution
include("wigner.jl")

# Integration schemes
include("time_evolution.jl")

# Ensemble
include("ensemble.jl")

# Phonons operations
include("phonons.jl")

# Interface with ASE
include("calculator.jl")

# External forces
include("external_f.jl")

# Core structures
export Dynamics, Ensemble, ElectricField, WignerDistribution

# Initialization functions
export init_from_dyn,
    init_ensemble_from_python, init_calculator, fake_field, equilibrium_ensemble

# Forces and electric fields
export generate_ensemble!,
    calculate_ensemble!, get_averages!, get_alphabeta, get_correlators
export get_average_forces, get_classic_forces
export pulse, sin_field, gaussian1, fake_dielectric_constant, read_charges_from_out!
export integrate!

end
