using AtomicSymmetries
using QuantumGaussianDynamics
using Test

using Unitful, UnitfulAtomic
using PyCall

@pyimport ase
@pyimport ase.calculators.emt as emt
@pyimport cellconstructor
@pyimport cellconstructor.Phonons as PH


function test_gold_dynamics()
    algorithm = "generalized-verlet"
    dt = 0.1u"fs"
    total_time = 10.0u"fs"
    dyn_file = joinpath(@__DIR__, "data/final_dyn")
    nqirr = 4
    N_configs = 100
    temperature = 300.0u"K"

    # We do not want ASR to be imposed
    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                seed = 1234,
                                                save_each=1)

    dyn = PH.Phonons(dyn_file, nqirr)
    wigner = QuantumGaussianDynamics.init_from_dyn(dyn, temperature, settings)

    # Get the symmetry group
    symmetry_group = get_symmetry_group_from_spglib(wigner)
    

    # Initialize the ensemble
    ensemble = QuantumGaussianDynamics.Ensemble(wigner_dist, settings; 
                                                n_configs=N_configs, 
                                                temperature=temperature)

    # Get the potential using EMT
    calc = emt.EMT()
    calc! = QuantumGaussianDynamics.init_calculator(calc, wigner, ase.Atoms)

    efield = QuantumGaussianDynamics.fake_field()

    QuantumGaussianDynamics.generate_ensemble!(N_configs, ensemble, wigner_dist)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, calc!)

    QuantumGaussianDynamics.integrate!(wigner_dist, ensemble, settings, calc!, efield;
                                       symmetry_group = symmetry_group)
end


if abspath(PROGRAM_FILE) == @__FILE__
    test_gold_dynamics()
end
