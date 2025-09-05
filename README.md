[![CI](https://github.com/NonequilibriumQuantumGaussianDynamics/QuantumGaussianDynamics.jl/actions/workflows/main.yml/badge.svg)](https://github.com/NonequilibriumQuantumGaussianDynamics/QuantumGaussianDynamics.jl/actions/workflows/main.yml)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://NonequilibriumQuantumGaussianDynamics.github.io/QuantumGaussianDynamics.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10%20|%201.11-9558B2)](https://julialang.org/downloads/)

# QuantumGaussianDynamics.jl

This package performs the quantum nuclear dynamics of a system under impulsive radiation (Pump-Probe).
It works by integrating the Time-Dependent Self-Consistent Harmonic Approximation equations of motion.

## Installation
To use it, you must activate the environment and install the dependencies.

```bash
julia --project=/path/to/QuantumGaussianDynamics
```

Then, instantiate the environment and install the dependencies.

```julia
using Pkg
Pkg.instantiate()
```

This will create a file named ``Manifest.toml`` containing the current state of the environment.

To employ the ASE interface for the calculators, you must install the following Python packages:
```bash
pip install ase
pip install cellconstructor
pip install python-sscha
````

## References

Details about the numerical methods can be found at 
> F. Libbi *et al.*, *Atomistic simulations of out-of-equilibrium quantum nuclear dynamics*, npj Computational Materials  11, 144 (2025) https://doi.org/10.1038/s41524-025-01588-4

