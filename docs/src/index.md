# QuantumGaussianDynamics.jl 
# Development Documentation

Simulate the quantum dynamics of a Gaussian wave packet by solving the equations of motion of the Time-Dependent Self-Consistent Harmonic Approximation (TD-SCHA).

The code is written in Julia for high performance.

Note: This documentation is under development and may be incomplete.

# Table of contents

```@contents
```

# Installation
To use this package, you must first install the **CellConstructor** module from the SSCHA code.  
> **Note:** For compatibility with CellConstructor, Python **3.10** is required.

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

Sometimes the default Python used by `PyCall` differs from the one where all the required packages are installed.  
You can check this with:

```
julia; using PyCall; PyCall.python
```

if the output is different from that of 

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

To run a TD-SCHA simulation, follow these steps:

- Set up the configuration variables: time step, total simulation time, number of configurations, etc.
- Load the initial conditions (e.g., the Gaussian distribution encoded by a dynamical matrix).
- Set up the calculator for interatomic energies and forces.
- Define the external force (electric field, etc.).
- Run the dynamics.

These steps are discussed in more detail in the following subsections.

## Setup the configuration variables

First, we need to provide the settings in a structure called `Dynamics`.  
This structure specifies the total simulation time, the time step, the number of configurations, and other integration parameters.

```julia
settings = QuantumGaussianDynamics.Dynamics(
    dt = 0.1,
    total_time = 10.0,
    algorithm = 'generalized-verlet',
    kong_liu_ratio = 1.0,
    verbose = true,
    evolve_correlators = true,
    save_filename = method,
    save_correlators = true,
    save_each = 1,
    N = 100,
    seed = 1254,
    correlated = true,
)
```

See [`Dynamics`](@ref QuantumGaussianDynamics.Dynamics) for more details.


### Load the initial conditions (dynamical matrix)

The TD-SCHA simulation starts from an initial equilibrium Gaussian distribution, obtained by solving the Stochastic Self-Consistent Harmonic Approximation (SSCHA).  
For details on how to obtain this solution, see the [SSCHA package](https://sscha.eu).

Here, we assume you already have a dynamical matrix and want to use it to initialize the TD-SCHA dynamics.  
To do this, we use `PyCall` to load the dynamical matrix with a Python calculator.

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
rho = init_from_dyn(dyn, temperature, settings)
```

See [`init_from_dyn`](@ref QuantumGaussianDynamics.init_from_dyn) for details.


## Initialize the forces calculator

The force calculator can be initialized using the ASE interface.
If `ase_calculator` is a valid PyObject representing an ASE calculator, the following code initializes the forces calculator.

```julia

crystal = QuantumGaussianDynamics.init_calculator(ase_calculator, rho, ase.Atoms)
```

# External force

External forces are introduced in the dynamics as (`ElectricField`)[@ref QuantumGaussianDynamics.ElectricField] object. 
One type of external perturbations is already implemented: `ElectricField`. There are several pulse shapes available in [`pulse`](@ref QuantumGaussianDynamics.pulse). Here is an example of a pulse with a [`Gaussian wave-packet`](@ref QuantumGaussianDynamics.pulse) shape.

```julia

# Equation of the pulse: E(t)=A*cos(2\pi*freq*t)*exp(-(t-t0)^2/(2*sig^2))
A = 3000.0 #kV/cm
freq = 2.4 #THz
t0 = 1875.0 #fs
sig = 468.0 #fs
edir = [0,0,1.0] 
field_fun = QuantumGaussianDynamics.pulse
field_f = t -> field_fun(t,A,freq,t0,sig)

```

# Run!

The dynamics is run throuhg the function [`integrate!`](@ref QuantumGaussianDynamics.integrate!)

## Index

```@index
```
