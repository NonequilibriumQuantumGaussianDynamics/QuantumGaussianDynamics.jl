# 📦 QuantumGaussianDynamics.jl

[![CI](https://github.com/NonequilibriumQuantumGaussianDynamics/QuantumGaussianDynamics.jl/actions/workflows/main.yml/badge.svg)](https://github.com/NonequilibriumQuantumGaussianDynamics/QuantumGaussianDynamics.jl/actions/workflows/main.yml)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://NonequilibriumQuantumGaussianDynamics.github.io/QuantumGaussianDynamics.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10%20|%201.11-9558B2)](https://julialang.org/downloads/)

*A Julia package for simulating quantum nuclear dynamics of materials under impulsive radiation, based on the Time-Dependent Self-Consistent Harmonic Approximation (TD-SCHA).*

---

## Features
- Integration of **TD-SCHA** equations of motion for nuclear quantum dynamics  
- Handles electric fields of custom shape and intensity  
- Interfaces with **ASE** and **CellConstructor** for calculator backends  

---

## Installation

To use this package, you must first install the CellConstructor module from the SSCHA code.
Note: For compatibility with CellConstructor, Python 3.10 is required.

```bash
conda create -n sscha -c conda-forge python=3.10 gfortran libblas lapack openmpi openmpi-mpicc pip numpy scipy spglib=2.2 setuptools=64
conda activate sscha
pip install ase mpi4py
pip install cellconstructor python-sscha tdscha
```

Clone the folder locally

```bash
git clone git@github.com:NonequilibriumQuantumGaussianDynamics/QuantumGaussianDynamics.jl.git
```

and instantiate the package

```bash
julia --project=/path/to/QuantumGaussianDynamics
```

```julia
using Pkg
Pkg.instantiate()
```

This will create a file named ``Manifest.toml`` contaning the current state of the environment.

Sometimes the default python used by PyCall is different with respect to the main one on which all the packages are installed.
This can be checked with

```julia
using PyCall; PyCall.python
```

if the output is different to that of

```bash
which python
```

then the pkg PyCall should be rebuild as

```julia
ENV["PYTHON"] = "[path to the right python]"
import Pkg
Pkg.build("PyCall")
```

## References

Details about the numerical methods can be found at 
> F. Libbi *et al.*, *Atomistic simulations of out-of-equilibrium quantum nuclear dynamics*, npj Computational Materials  11, 144 (2025) https://doi.org/10.1038/s41524-025-01588-4

while the thoeretical formulation is conteined in   

> L. Monacelli *et al.*, *Time-dependent self-consistent harmonic approximation: Anharmonic nuclear quantum
dynamics and time correlation functions*, Physical Review B 103, 104305 (2021) https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.104305

> A. Siciliano *et al.*, *Wigner Gaussian dynamics: Simulating the anharmonic and quantum ionic motion*, Physical Review B 107, 174307 (2023) https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.174307
