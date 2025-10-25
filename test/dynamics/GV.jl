#import Pkg
#Pkg.activate("/Users/flibbi/Documents/programs/QuantumGaussianDynamics.jl")
using QuantumGaussianDynamics
#@info "Using package at" path = pathof(QuantumGaussianDynamics)

using MPI
using Test

using PyCall
using LinearAlgebra
using DelimitedFiles

@pyimport ase
@pyimport ase.calculators.emt as emt

MPI.Init()

@testset "Dynamics of H2 molecule" begin

    # Load the dyn corresponding to the equilibrium structure of a SSCHA calculation
    TEMPERATURE = 0.0
    dyn_file = joinpath(@__DIR__, "final_result")
    ens_file = joinpath(@__DIR__, "sscha_ensemble")
    ens_bin = 1
    ndyn = 1
    py_ensemble, dyn = equilibrium_ensemble(TEMPERATURE, dyn_file, ens_file, ndyn, ens_bin)

    method = "generalized-verlet" # use this one
    settings = Dynamics(
        dt = 0.1,
        total_time = 10.0,
        algorithm = method,
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
    rho = init_from_dyn(dyn, TEMPERATURE, settings)
    ensemble = init_ensemble_from_python(py_ensemble, settings)


    # Specify here the ASE calculator
    calculator = emt.EMT()
    crystal = init_calculator(calculator, rho, ase.Atoms)


    # Electric field
    # If you do not want to apply any field, use fake_field, like this. Otherwise, prepare a fake ph.out to read the effective charges and the dielectric constant
    efield = fake_field(rho.n_atoms)

    # Displacement from equilbrium, optional
    rho.P_av[1] += 0.01 #sqrt(Ry)

    # Initialize forces
    generate_ensemble!(settings.N, ensemble, rho)
    calculate_ensemble!(ensemble, crystal)
    get_average_forces(ensemble)
    get_classic_forces(rho, crystal)

    # Run!
    @time integrate!(rho, ensemble, settings, crystal, efield)

    data = readdlm(method*"0.1-10.0-100.pos")
    ref = readdlm(joinpath(@__DIR__, "reference_generalized-verlet0.1-10.0-100.pos"))

    @test norm(data .- ref) < 1e-8
end
