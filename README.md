# Quantum Gaussian Dynamics

This package perform the quantum dynamics of a Wigner Gaussian distribution solving the
equation of motion of the Time-Dependent Self-Consistent Harmonic Approximation


The code is written in julia to achieve high performance.


# Activate and add dependencies
open Julia shell
enter the package mode (type ])
activate the packacge (activate . )
add the dependency (add LinearAlgebra)


# Install on a new machine
julia
using Pkg
#Pkg.activate("path/QuantumGaussianDynamics.jl")
#Pkg.instantiate()
Pkg.add(PackageSpec(path="path/QuantumGaussianDynamics.jl"))

# Also install cellconstructor and sscha
pip install Cellconstructor
pip install python-sscha
