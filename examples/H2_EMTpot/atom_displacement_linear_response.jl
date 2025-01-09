using QuantumGaussianDynamics
using DelimitedFiles
using LinearAlgebra
using Unitful, UnitfulAtomic

using PyCall

# Load the python modules of the SSCHA
# To setup the equilibrium state of the H2 molecule
@pyimport cellconstructor as CC 
@pyimport cellconstructor.Phonons as PH
@pyimport sscha
@pyimport sscha.Ensemble as PyEnsemble
@pyimport ase
@pyimport ase.calculators.emt as emt


# Run the calculation
function main()
    method = "generalized-verlet"
    fname = "linear_responce_momentum_pert_noclean"
    dt = 0.1u"fs"
    t_max = 20.0u"fs"
    temperature = 0.0u"K"



    # Load the SSCHA dynamical matrix
    # @__DIR__ is the directory of the current script
    sscha_path = @__DIR__
    dyn = PH.Phonons(joinpath(sscha_path, "sscha_ensemble/dyn_gen_pop1_"), 1)

    # Load the converged SSCHA ensemble
    py_ensemble = PyEnsemble.Ensemble(dyn, ustrip(temperature))
    py_ensemble.load_bin(joinpath(sscha_path, "sscha_ensemble"), 1)
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()

    t1 = time()


    """
    Inputs
    dt: discretization time, femtoseconds
    t_max: run time, femtoseconds
    algorithm: evolution algorithm
    kong_liu_ratio: limit of the ensemble stochastic criterion
    evolve_correlators: 
    N: number of configurations
    seed: seed for the calculations. If you do not want to specify any seed, use seed=0
    correlated: whether the dynamics is correlated or not. Avoid with a kong-liu lower than 1. 
    """

    # Properties of the simulation
    kong_liu_ratio = 0.2
    N_configs = py_ensemble.N

    settings = QuantumGaussianDynamics.Dynamics(dt, t_max, N_configs; algorithm = method, kong_liu_ratio = kong_liu_ratio, 
                                               verbose = true,  evolve_correlators = true, save_filename = fname, 
                                              save_correlators = false, save_each = 1, seed=1254, correlated = false)

    # Prepare the gradient cleaning
    #set_clean_gradients!(settings, true)

    rho = QuantumGaussianDynamics.init_from_dyn(dyn, temperature, settings)
    ensemble = QuantumGaussianDynamics.init_ensemble_from_python(py_ensemble, settings)

    # Add the symmetries
    symmetry_group = get_symmetry_group_from_spglib(rho)
    println("Number of symmetries: ", length(symmetry_group.symmetries))

    # Specify here the ASE calculator
    calculator = emt.EMT()
    crystal = QuantumGaussianDynamics.init_calculator(calculator, rho, ase.Atoms)

    # Electric field
    # If you do not want to apply any field, use fake_field, like this. Otherwise, prepare a fake ph.out to read the effective charges and the dielectric constant
    efield = QuantumGaussianDynamics.fake_field(get_natoms(rho))

    # Add an initial momentum
    rho.P_av[1] = 0.01
    rho.P_av[4] = -0.01

    # Run!
    QuantumGaussianDynamics.integrate!(rho, ensemble, settings, crystal, efield;
                                      symmetry_group = symmetry_group)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end



