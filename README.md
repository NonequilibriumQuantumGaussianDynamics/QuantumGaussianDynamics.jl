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

