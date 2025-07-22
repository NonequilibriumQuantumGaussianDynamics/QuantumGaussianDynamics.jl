push!(LOAD_PATH, "/home/flibbi/programs/sscha/QuantumGaussianDynamics.jl")   # Needed for finding the installation of the code
#println(LOAD_PATH)

using MPI
using Test
import QuanumGaussianDynamics
using QuanumGaussianDynamics.QuanumGaussianDynamics

using PyCall
using LinearAlgebra
using DelimitedFiles

@pyimport cellconstructor.Phonons as PH
@pyimport cellconstructor as CC
@pyimport  sscha.Ensemble as  PyEnsemble
@pyimport ase
@pyimport ase.calculators.emt as emt

MPI.Init()

@testset "Dynamics of H2 molecule" begin

    # Load the dyn corresponding to the equilibrium structure of a SSCHA calculation
    TEMPERATURE = 0.0 
    sscha_path = "./"
    dyn = PH.Phonons.(sscha_path * "final_result", 1)
    py_ensemble = PyEnsemble.Ensemble(dyn, TEMPERATURE)
    py_ensemble.load_bin(sscha_path * "sscha_ensemble", 1)
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()

    method = "semi-implicit-verlet" # use this one
    settings = QuanumGaussianDynamics.Dynamics(dt = 0.1, total_time = 10.0, algorithm = method, kong_liu_ratio = 1.0, 
                                               verbose = true,  evolve_correlators = true, save_filename = method, 
                                              save_correlators = true, save_each = 1, N=100,seed=1254, correlated = true)
    rho = QuanumGaussianDynamics.init_from_dyn(dyn, Float64(TEMPERATURE), settings)
    ensemble = QuanumGaussianDynamics.init_ensemble_from_python(py_ensemble, settings)

    # Initialization
    dv_dr = zeros(rho.n_atoms*3)
    d2v_dr2 = zeros(rho.n_atoms*3,rho.n_atoms*3)
    QuanumGaussianDynamics.get_averages!(dv_dr, d2v_dr2, ensemble, rho)

    # Specify here the ASE calculator
    calculator = emt.EMT()
    crystal = QuanumGaussianDynamics.init_calculator(calculator, rho, ase.Atoms)


    # Electric field
    # If you do not want to apply any field, use fake_field, like this. Otherwise, prepare a fake ph.out to read the effective charges and the dielectric constant
    efield = QuanumGaussianDynamics.fake_field(rho.n_atoms)

    # Displacement from equilbrium, optional
    rho.P_av[1] += 0.01 #sqrt(Ry)

    # Some calculation
    QuanumGaussianDynamics.generate_ensemble!(settings.N,ensemble, rho)
    QuanumGaussianDynamics.calculate_ensemble!(ensemble, crystal)
    QuanumGaussianDynamics.get_average_forces(ensemble)
    QuanumGaussianDynamics.get_classic_forces(rho,crystal)

    # Run!
    QuanumGaussianDynamics.integrate!(rho, ensemble, settings, crystal, efield )

    data = readdlm(method*"0.1-10.0-100.pos")
    ref = readdlm("../../examples/H2/semi-implicit-verlet0.1-10.0-100.pos")

    @test norm(data.-ref) < 1e-8
end
