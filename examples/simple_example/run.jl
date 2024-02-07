t0 = time()
# push!(LOAD_PATH, "/home/flibbi/programs/sscha/QuantumGaussianDynamics.jl")   # Needed for finding the installation of the code
# println(LOAD_PATH)

using MPI
MPI.Init()
import QuanumGaussianDynamics
using QuanumGaussianDynamics.QuanumGaussianDynamics

using PyCall
using LinearAlgebra
using DelimitedFiles

@pyimport cellconstructor.Phonons as PH
@pyimport cellconstructor as CC
@pyimport  sscha.Ensemble as  PyEnsemble
@pyimport ase
@pyimport quippy.potential as potential
@pyimport DIMER.Calculator as DIMERCalc


# Load the dyn corresponding to the equilibrium structure of a SSCHA calculation
TEMPERATURE = 0.0 
sscha_path = "./"
dyn = PH.Phonons.(sscha_path * "final_result", 1)
py_ensemble = PyEnsemble.Ensemble(dyn, TEMPERATURE)
py_ensemble.load_bin(sscha_path * "sscha_ensemble", 1)
dyn.Symmetrize()
dyn.ForcePositiveDefinite()

t1 = time()


"""
Inputs
dt: discretization time, femtoseconds
total_time: run time, femtoseconds
algorithm: evolution algorithm
kong_liu_ratio: use 1.0, lower values may not conserve the energy
evolve_correlators: use true
N: number of configurations
seed: seed for the calculations. If you do not want to specify any seed, use seed=0
correlated: whether the dynamics is correlated or not. 
"""
method = "semi-implicit-verlet" # use this one
settings = QuanumGaussianDynamics.Dynamics(dt = 1.0, total_time = 500.0, algorithm = method, kong_liu_ratio = 1.0, 
                                           verbose = true,  evolve_correlators = true, save_filename = method, 
                                          save_correlators = true, save_each = 1, N=1000,seed=1254, correlated = true)
rho = QuanumGaussianDynamics.init_from_dyn(dyn, Float64(TEMPERATURE), settings)
ensemble = QuanumGaussianDynamics.init_ensemble_from_python(py_ensemble, settings)

# Initialization
dv_dr = zeros(rho.n_atoms*3)
d2v_dr2 = zeros(rho.n_atoms*3,rho.n_atoms*3)
QuanumGaussianDynamics.get_averages!(dv_dr, d2v_dr2, ensemble, rho)

# Specify here the ASE calculator
calculator = DIMERCalc.DIMERCalculator(2, case= "quartic", k2 = -3.0, k3 = 12.0)
crystal = QuanumGaussianDynamics.init_calculator(calculator, rho, ase.Atoms)


# Electric field
# If you do not want to apply any field, use fake_field, like this. Otherwise, prepare a fake ph.out to read the effective charges and the dielectric constant
# efield = QuanumGaussianDynamics.fake_field(rho.n_atoms)

# If you want to apply the pulse, specify the parameters
# Equation of the pulse: E(t)=A*cos(2\pi*freq*t)*exp(-(t-t0)^2/(2*sig^2))
A = 3000.0 # 3000.0 #kV/cm
freq = 2.4 #THz
t0 = 1875.0 #fs
sig = 468.0 #fs
edir = [0,0,1.0] #Polarization of the field, must have norm=1
field_fun = deepcopy(QuanumGaussianDynamics.pulse)
field_f = t -> field_fun(t,A,freq,t0,sig)

# Read effective charges and dielectric tensor from ph.out
Zeff, eps = QuanumGaussianDynamics.read_charges_from_out!("ph.out",  rho)
efield = QuanumGaussianDynamics.ElectricField(fun = field_f, Zeff = Zeff, edir=edir, eps = eps)

# Some calculation
QuanumGaussianDynamics.generate_ensemble!(settings.N,ensemble, rho)
QuanumGaussianDynamics.calculate_ensemble!(ensemble, crystal)
QuanumGaussianDynamics.get_average_forces(ensemble)
QuanumGaussianDynamics.get_classic_forces(rho,crystal)

# Run!
QuanumGaussianDynamics.integrate!(rho, ensemble, settings, crystal, efield )

