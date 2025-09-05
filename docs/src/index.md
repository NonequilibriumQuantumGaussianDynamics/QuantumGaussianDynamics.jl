# QuantumGaussianDynamics.jl Development Documentation

Perform the quantum dynamics of a Gaussian wave-packet solving the equation of motion of the Time-Dependent Self-Consistent Harmonic Approximation.

The code is written in julia to achieve high performance.

Note that this documentation is under development and may not be complete.

# Table of contents

```@contents
```

# Installation
To use it, you must first install the cellconstructor routing from the sscha code. NB for compatibility
issues with the cellconstructor code, it is necessary to install Python 3.10.

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

```
julia; using PyCall; PyCall.python
```

if the output is different to that of 

```
which python
```

then the pkg PyCall should be rebuild as

```
ENV["PYTHON"] = "[path to the right python]"
import Pkg
Pkg.build("PyCall")
```


# Simulation setup

To run a TD-SCHA simulation, we need the following steps:

- Setup the configuration variables: time-step, total simulation time, number of configurations, etc.
- Load the initial conditions (e.g. the Gaussian distribution encoded via a dynamical matrix)
- Setup the calculator for interatomic energies and forces
- Setup the external force (electric field, etc)
- Setup the symmetries of the system + external force (optional)
- Run the dynamics

These steps will be discussed in details in the next subsections

## Setup the configuration variables

To perform a dynamics, we need a configuration variables called `Dynamics`.
Here we specify the total simulation time, time-step and number of configurations, as well as other properties of the integration.

In the following, we show the available functions to setup the dynamics.


### Load the initial conditions (dynamical matrix)

The TD-SCHA evolves starting from a initial equilibrium Gaussian distribution, obtained by solving the Stochastic Self-Consistent Harmonic Approximation (SSCHA).
For details on how to obtain this solution, see the [SSCHA package](https://sscha.eu).

Here, we assume you already obtained a dynamical matrix and want to use it to initialize the TD-SCHA dynamics.

For this, we use PyCall to load the dynamics using a python calculator

```julia
using PyCall
cc = pyimport("cellconstructor")
PH = pyimport("cellconstructor.Phonons")

# Load the dynamical matrix using cellconstructor from python
dyn = PH.Phonons("dynfile", nqirr=1)
```

For more details on how to load the dynamical matrix, see the [cellconstructor documentation](https://sscha.eu/documentation/).

The dynamical matrix as a python object can be converted into a Wigner distribution for TD-SCHA using the following function

```julia
using QuantumGaussianDynamics

temperature = 300.0 #K

# Convert the python dynamical matrix into a Wigner distribution
wigner = init_from_dyn(dyn, temperature, settings)
```

In the following, the API for the function `init_from_dyn` is shown

```@docs
QuantumGaussianDynamics.init_from_dyn
```

## Initialize the forces calculator

The force calculator can be initialized using the ASE interface.
If `ase_calculator` is a valid PyObject representing an ASE calculator, the following code initializes the forces calculator.

# External force

External forces are introduced in the dynamics as `ExternalPerturbation` objects. The code introduces few kinds of such perturbations, like IR electric fields, Raman lasers, etc. The important thing is that the perturbation must be a subtype of ExternalPerturbation and must implement the method `get_external_forces` defined as

```julia
get_external_forces(time :: T, perturbation :: ExternalPerturbation, wigner :: WignerDistribution{T}) :: Vector{T} where {T}
```

It must return a vector of size `n_dims * n_atoms`, with the external force acting on each atom at the given time. All quantities must be expressed in Hartree atomic units.

Two types of external perturbations are already implemented: `ElectricField` and `StimulatedRamanField`.



## Index

```@index
```
