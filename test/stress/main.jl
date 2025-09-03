#import Pkg
#Pkg.activate("/Users/flibbi/Documents/programs/QuantumGaussianDynamics.jl")
using QuantumGaussianDynamics
#@info "Using package at" path = pathof(QuantumGaussianDynamics)

using MPI
using Test

using PyCall
using LinearAlgebra
using DelimitedFiles

@pyimport cellconstructor.Phonons as PH
@pyimport cellconstructor as CC
@pyimport  sscha.Ensemble as  PyEnsemble
@pyimport ase
@pyimport ase.calculators.emt as emt

MPI.Init()

@testset "Stress computation" begin

    # Load the dyn corresponding to the equilibrium structure of a SSCHA calculation
    TEMPERATURE = 0.0 
    sscha_path = "./"
    dyn = PH.Phonons.(joinpath(@__DIR__, "final_result"), 1)
    py_ensemble = PyEnsemble.Ensemble(dyn, TEMPERATURE)
    py_ensemble.load_bin(joinpath(@__DIR__, "sscha_ensemble"), 1)
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()

    method = "semi-implicit-verlet" # use this one
    settings = QuantumGaussianDynamics.Dynamics(dt = 0.1, total_time = 10.0, algorithm = method, kong_liu_ratio = 1.0, 
                                               verbose = true,  evolve_correlators = true, save_filename = method, 
                                              save_correlators = true, save_each = 1, N=100,seed=1254, correlated = true)
    rho = QuantumGaussianDynamics.init_from_dyn(dyn, Float64(TEMPERATURE), settings)
    ensemble = QuantumGaussianDynamics.init_ensemble_from_python(py_ensemble, settings)

    # Initialization
    dv_dr = zeros(rho.n_atoms*3)
    d2v_dr2 = zeros(rho.n_atoms*3,rho.n_atoms*3)
    QuantumGaussianDynamics.get_averages!(dv_dr, d2v_dr2, ensemble, rho)

    # Specify here the ASE calculator
    calculator = emt.EMT()
    crystal = QuantumGaussianDynamics.init_calculator(calculator, rho, ase.Atoms)


    # Electric field
    # If you do not want to apply any field, use fake_field, like this. Otherwise, prepare a fake ph.out to read the effective charges and the dielectric constant
    efield = QuantumGaussianDynamics.fake_field(rho.n_atoms)

    # Displacement from equilbrium, optional
    rho.R_av[1] += 0.01 #sqrt(Ry)
    rho.R_av[5] += 0.01 #sqrt(Ry)

    # Some calculation
    QuantumGaussianDynamics.generate_ensemble!(settings.N,ensemble, rho)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, crystal)
    stress = QuantumGaussianDynamics.get_average_stress(ensemble, rho)
    #writedlm("stress.dat", stress')
    expected = readdlm("stress.dat")

    @test stress ≈ vec(expected) atol=1e-8 rtol=1e-8
    
end
