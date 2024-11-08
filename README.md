# Quantum Gaussian Dynamics

This package perform the quantum dynamics of a Wigner Gaussian distribution solving the
equation of motion of the Time-Dependent Self-Consistent Harmonic Approximation.


The code is written in julia to achieve high performance.

## Installation
To use it, you must activate the environment and install the dependencies.

```bash
julia --project=/path/to/QuantumGaussianDynamics
```

Then instantiate the environment and install the dependencies.

```julia
using Pkg
Pkg.instantiate()
```

This will create a file named ``Manifest.toml`` contaning the current state of the environment.

To employ the ASE interface for the calculators, you must install the following python packages:
```bash
pip install ase
pip install cellconstructor
pip install python-sscha
```


# TODO

- [ ] Implement a test with a fully anharmonic SSCHA calculator.
