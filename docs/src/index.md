# QuantumGaussianDynamics.jl Development Documentation

Perform the quantum dynamics of a Gaussian wave-packet solving the equation of motion of the Time-Dependent Self-Consistent Harmonic Approximation.

The code is written in julia to achieve high performance.

Note that this documentation is under development and may not be complete.
The code is not yet registered or released as open-source package. If you want to employ it for your research, please contact [the author](mailto:lorenzo.monacelli@uniroma1.it).

# Table of contents

```@contents
```

# Installation
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
Here an example using `Unitful.jl` to define units

```jldoctest
julia> using Unitful, QuantumGaussianDynamics

julia> time_step = 1.0u"fs"
1.0 fs

julia> total_time = 1.0u"ps"
1.0 ps

julia> N_configs = 1000
1000

julia> dynamics = Dynamics(time_step, total_time, N_configs)
Dynamics{Float64}(1.0, 1000.0, "generalized-verlet", 1.0, true, true, 0, 1000, true, ASR{Float64}(false, 1.0e-8, 3), "dynamics", false, 100)
```

In the following, we show the available functions to setup the dynamics.

```@docs
Dynamics
```

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
using QuantumGaussianDynamics, Unitful

temperature = 300.0u"K"

# Convert the python dynamical matrix into a Wigner distribution
wigner = init_from_dyn(dyn, temperature, settings)
```

In the following, the API for the function `init_from_dyn` is shown

```@docs
init_from_dyn
```

## Initialize the forces calculator

The force calculator can be initialized using the ASE interface.
If `ase_calculator` is a valid PyObject representing an ASE calculator, the following code initializes the forces calculator.

```julia
using QuantumGaussianDynamics
using PyCall
ATM = pyimport("ase.atoms")

calculator = QuantumGaussianDynamics.init_calculator(ase_calculator, wigner, ATM.Atoms)
```

Alternatively, much more efficiently Julia function can be used to calculate the forces.

```@docs
QuantumGaussianDynamics.init_calculator
```

If we chose to use a julia function, the `init_calculator` is not necessary, and we can replace the resulting `calculator` object in the final dynamics with the julia function that inplaces modifies the forces like

```julia
function force_calculator!(forces :: AbstractVector{T}, stress :: AbstractVector{T}, coords :: AbstractVector{T}) :: T where {T}
    # Inplace calculation of the forces
    # Calculation of the energy
    return energy
end
```

The function `force_calculator!` takes 1D vectors (flattened `3 * N_atoms` arrays) and returns the energy of the system. Units are assumed in Hartree Atomic Units.
The stress is a 6-component vector representing the stress tensor in Voigt notation.


# External force

External forces are introduced in the dynamics as `ExternalPerturbation` objects. The code introduces few kinds of such perturbations, like IR electric fields, Raman lasers, etc. The important thing is that the perturbation must be a subtype of ExternalPerturbation and must implement the method `get_external_forces` defined as

```julia
get_external_forces(time :: T, perturbation :: ExternalPerturbation, wigner :: WignerDistribution{T}) :: Vector{T} where {T}
```

It must return a vector of size `n_dims * n_atoms`, with the external force acting on each atom at the given time. All quantities must be expressed in Hartree atomic units.

Two types of external perturbations are already implemented: `ElectricField` and `StimulatedRamanField`.

```@docs
ElectricField
StimulatedRamanField
```

The stimulated Raman can be created as
```@docs
QuantumGaussianDynamics.get_impulsive_raman_pump
```


# Troubleshooting 

- When running an example that uses an ``ase`` calculator, I get the following error:

```bash
Fatal error in internal_Comm_size: Invalid communicator, error stack:
internal_Comm_size(30769): MPI_Comm_size(comm=0x18d45d20, size=0x7ffcc56f0ddc) failed
internal_Comm_size(30723): Invalid communicator
[unset]: PMIU_write error; fd=-1 buf=:cmd=abort exitcode=1007251461 message=Fatal error in internal_Comm_size: Invalid communicator, error stack:
internal_Comm_size(30769): MPI_Comm_size(comm=0x18d45d20, size=0x7ffcc56f0ddc) failed
internal_Comm_size(30723): Invalid communicator
:
system msg for write_line failure : Bad file descriptor
```

This error is due to the mismatch between the version of MPI linked by the julia and python libraries MPI.jl and mpi4py.
In particular, ``ase`` always forces to load ``mpi4py`` causing the error to pop out. To solve, you can either reinstall mpi4py to 
match the same MPI version as MPI.jl, or exploit MPIPreferences.jl by julia to force it to use the same version as mpi4py.
In case ase is installed on a virtual environment, do the following steps:

```julia
julia> using MPIPreferences
julia> MPIPreferences.use_system_binary(; extra_path=["path/to/environment/lib", "path/to/environment/bin"])
```

This will generate a file ``LocalPreferences.toml``. Place it in the root of your project folder, then reinstantiate the 
environment with

```julia
julia> using Pkg
julia> Pkg.instantiate()
```

To setup this for a global julia installation and further information, 
see ``MPIPreferences.jl`` [documentation](https://juliaparallel.org/MPI.jl/stable/configuration/)


## Index

```@index
```
