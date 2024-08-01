t0 = time()
# push!(LOAD_PATH, "/home/flibbi/programs/sscha/QuantumGaussianDynamics.jl")
# println(LOAD_PATH)
using QuantumGaussianDynamics

using PyCall
using LinearAlgebra
using DelimitedFiles


# Python dependencies
@pyimport cellconstructor.Phonons as PH
@pyimport cellconstructor as CC
@pyimport  sscha.Ensemble as  PyEnsemble
@pyimport ase
@pyimport quippy.potential as potential
@pyimport DIMER.Calculator as DIMERCalc

#Constants to be moved for later use
#CONV_BOHR = 1.88972598

TEMPERATURE = 0.0 #FLOAT

# Load the dyn corresponding to the equilibrium structure of a SSCHA calculation
sscha_path = "./"
dyn = PH.Phonons.(sscha_path * "final_result", 1)
py_ensemble = PyEnsemble.Ensemble(dyn, TEMPERATURE)
py_ensemble.load_bin(sscha_path * "sscha_ensemble", 1)
dyn.Symmetrize()
dyn.ForcePositiveDefinite()

t1 = time()


# Initialization
method = "semi-implicit-euler"
method = "none"
settings = QuantumGaussianDynamics.Dynamics(dt = 0.1, total_time = 50.0, algorithm = method, kong_liu_ratio = 1.0, 
                                           verbose = true,  evolve_correlators = true, save_filename = method, 
                                          save_correlators = true, save_each = 1, N=400)
rho = QuantumGaussianDynamics.init_from_dyn(dyn, Float64(TEMPERATURE), settings)

""" Initializaiton of the ensemble. TODO:
correctly subtract the translations
correctly reduce the number of modes by 3
load the ensamble through python and pass the info to the julia structure
add check for positive frequencies
check units
"""
ensemble = QuantumGaussianDynamics.init_ensemble_from_python(py_ensemble, settings)

QuantumGaussianDynamics.update_weights!(ensemble, rho)
QuantumGaussianDynamics.get_average_energy(ensemble)
QuantumGaussianDynamics.get_average_forces(ensemble)
writedlm("weights.txt", ensemble.weights, ' ')
dv_dr = zeros(rho.n_atoms*3)
d2v_dr2 = zeros(rho.n_atoms*3,rho.n_atoms*3)
QuantumGaussianDynamics.get_averages!(dv_dr, d2v_dr2, ensemble, rho)

calculator = DIMERCalc.DIMERCalculator(2, case= "cubic", k2 = 0.1)
crystal = QuantumGaussianDynamics.init_calculator(calculator, rho, ase.Atoms)

#println("initial free energy", QuantumGaussianDynamics.get_average_energy(ensemble))
#println("initial forces", QuantumGaussianDynamics.get_average_forces(ensemble) )
# Display atoms
rho.P_av[1] += 0.01 #sqrt(Ry)
QuantumGaussianDynamics.generate_ensemble!(200,ensemble, rho)
QuantumGaussianDynamics.calculate_ensemble!(ensemble, crystal)
#println("free energy", QuantumGaussianDynamics.get_average_energy(ensemble))
#println("initial forces")
display(QuantumGaussianDynamics.get_average_forces(ensemble))
QuantumGaussianDynamics.get_classic_forces(rho,crystal)

QuantumGaussianDynamics.integrate!(rho, ensemble, settings, crystal )



