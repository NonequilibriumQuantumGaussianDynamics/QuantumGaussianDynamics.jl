using AtomicSymmetries
using QuantumGaussianDynamics
using Test

using Unitful, UnitfulAtomic
using PyCall

@pyimport ase
@pyimport ase.calculators.emt as emt
@pyimport cellconstructor
@pyimport cellconstructor.Phonons as PH
@pyimport spglib


function test_symmetric_anharmonic_dynamics()
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    ase_calc_module = pyimport("test_ase_calculator")

    algorithm = "generalized-verlet"
    dt = 1.0u"fs"
    total_time = 100.0u"fs"
    dyn_file = joinpath(@__DIR__, "data/dyn_harmonic_sym_converged")
    nqirr = 1
    N_configs = 100
    temperature = 0.0u"K"

    # We do not want ASR to be imposed
    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                seed = 1234,
                                                save_filename = "sym",
                                                save_each=1)

    dyn = PH.Phonons(dyn_file, nqirr)
    wigner = QuantumGaussianDynamics.init_from_dyn(dyn, temperature, settings)
    #wigner.R_av[1] += 0.001 * √(wigner.masses[1])

    # Get the symmetry group
    symmetry_group = get_symmetry_group_from_spglib(wigner; spglib_py_module=spglib)
    #symmetry_group = get_empty_symmetry_group(Float64)
    
    println("Number of symmetries: ", length(symmetry_group.symmetries))

    # Initialize the ensemble
    ensemble = QuantumGaussianDynamics.Ensemble(wigner, settings; 
                                                n_configs=N_configs, 
                                                temperature=temperature)

    # Get the potential using EMT
    k1 = 0.919347513/5 # 0.01892175 
    k2 = k1
    k3 = k1
    calc = ase_calc_module.Anharmonic3D_ASR(k1, k2, k3, 1.0, 1.0, 1.0, 0.1)
    calc! = QuantumGaussianDynamics.init_calculator(calc, wigner, ase.Atoms)

    efield = QuantumGaussianDynamics.fake_field(get_natoms(wigner))

    QuantumGaussianDynamics.generate_ensemble!(N_configs, ensemble, wigner)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, calc!)

    RR_corr_start = copy(wigner.RR_corr)

    QuantumGaussianDynamics.integrate!(wigner, ensemble, settings, calc!, efield;
                                       symmetry_group = symmetry_group)

    nat = get_natoms(wigner)
    for i in 1:3nat
        for j in 1:3nat
            @test wigner.RR_corr[i, j] ≈ RR_corr_start[i, j] atol=1e-8 rtol=5e-2
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    test_symmetric_anharmonic_dynamics()
    #test_gold_dynamics()
end
