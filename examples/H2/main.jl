using MPI
MPI.Init()
import QuantumGaussianDynamics
using QuantumGaussianDynamics.QuantumGaussianDynamics

using PyCall
using LinearAlgebra
using DelimitedFiles

@pyimport cellconstructor.Phonons as PH
@pyimport cellconstructor as CC
@pyimport sscha.Ensemble as PyEnsemble
@pyimport ase
@pyimport ase.calculators.emt as emt


#1) Load the equilibrium structure and ensemble
TEMPERATURE = 0.0
dyn_file = "final_result"
ens_file = "sscha_ensemble"
ens_bin = 1
ndyn = 1
py_ensemble, dyn = equilibrium_ensemble(TEMPERATURE, dyn_file, ens_file, ndyn, ens_bin)


#2) Dynamics settings
method = "generalized-verlet" # most accurate
settings = Dynamics(
    dt = 0.1,                     # time step, fs
    total_time = 10.0,            # total simulation time, fs
    algorithm = method,           # integration scheme
    kong_liu_ratio = 1.0,         # kong-liu ratio, reweighting
    verbose = true,               # verbosity of the output
    evolve_correlators = true,    # evolve <RR>, <PP>, <RP> matrices
    save_filename = method,       # filename
    save_correlators = true,      # save the full output
    save_each = 1,                # output step 
    N = 100,                      # number of stochastic configurations
    seed = 1254,                  # seed for reproducibility (0 = no seed)
    correlated = true,            # correlated approach
)
rho = init_from_dyn(dyn, TEMPERATURE, settings)
ensemble = init_ensemble_from_python(py_ensemble, settings)


# 3) ASE calculator
calculator = emt.EMT()
crystal = init_calculator(calculator, rho, ase.Atoms)


# 4) Electric field
A = 3000.0 #kV/cm 
freq = 2.4 #THz
t0 = 1875.0 #fs
sig = 468.0 #fs
edir = [0,0,1.0] 
field_fun = pulse(A, freq, t0, sig)
Zeff, eps = fake_dielectric_constant(rho.n_atoms)

efield = QuantumGaussianDynamics.ElectricField(fun = field_fun, Zeff = Zeff, edir=edir, eps = eps)

# 5) Run!
integrate!(rho, ensemble, settings, crystal, efield)
