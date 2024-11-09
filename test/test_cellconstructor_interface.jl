using QuantumGaussianDynamics
using PyCall
using Unitful, UnitfulAtomic
using Test
using PhysicalConstants.CODATA2018

const ω1 = 500.0u"c/cm" * 2π
const ω2 = 700.0u"c/cm" * 2π
const ω3 = 900.0u"c/cm" * 2π

"""
This test loads a dynamical matrix already converged within the SSCHA.
(Script in test_asr_scha.py)
And tries to run a dynamics with the same calculator.
If everything works correctly, the solution must be static.

This test works only if cellconstructor is installed.
"""
function test_dyn_scha_converged()
    # Allow importing a python module in the same directory as the test
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    ase_calc_module = pyimport("test_ase_calculator")

    # Import python sscha
    CC_py = pyimport("cellconstructor")
    PH_py = pyimport("cellconstructor.Phonons")
    calc_py_module = pyimport("test_ase_calculator")
    ATOMS_py = pyimport("ase.atoms")

    dt = 1.0u"fs"
    total_t = 200.0u"fs"
    N_configs = 1000
    algorithm = "generalized-verlet"
    temperature = 0.0u"K"

    # Setup the standard dynamics settings
    settings = QuantumGaussianDynamics.Dynamics(dt, total_t, N_configs;
                                                algorithm = algorithm,
                                                seed = 1234,
                                                save_each = 1,
                                                save_filename = "py_test")

    # Load the dynamical matrix
    dyn_py = PH_py.Phonons(joinpath(@__DIR__, "dyn_harmonic_converged"))

    # Convert from python
    wigner = QuantumGaussianDynamics.init_from_dyn(dyn_py, temperature, settings)

    # Prepare the ASE calculator
    # We use the same input as the python minimization
    k1 = 0.4596737565
    k2 = 0.6619319097037785
    k3 = 0.900960805409125
    calc_py = calc_py_module.Harmonic3D_ASR(k1, k2, k3)
    calc_jl = QuantumGaussianDynamics.init_calculator(calc_py, wigner, ATOMS_py.Atoms)

    # Prepare a fake field for the dynamics
    efield = QuantumGaussianDynamics.fake_field(wigner.n_atoms)

    # Prepare the ensemble
    ensemble = QuantumGaussianDynamics.Ensemble(wigner, settings; 
                                                n_configs= N_configs,
                                                temperature = temperature)

    # Generate the ensemble
    QuantumGaussianDynamics.generate_ensemble!(ensemble, wigner)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, calc_jl)
    
    # Save the starting values 
    starting_RR_corr = copy(wigner.RR_corr)
    starting_R_pos = copy(wigner.R_av)

    # Integrate the equation of motion (run the dynamics)
    QuantumGaussianDynamics.integrate!(wigner, ensemble, settings, calc_jl, efield)

    # Check that the final values are the same as the initial ones
    n_dims = size(wigner.RR_corr, 1)
    for i in 1:n_dims
        for j in 1:n_dims
            if abs(starting_RR_corr[i, j]) > 1e-4
                @test wigner.RR_corr[i,j] ≈ starting_RR_corr[i,j] rtol = 1e-1
            else 
                @test wigner.RR_corr[i,j] ≈ starting_RR_corr[i,j] atol = 1e-4
            end
        end
        if abs(starting_R_pos[i]) > 1e-4
            @test wigner.R_av[i] ≈ starting_R_pos[i] rtol = 1e-1
        else
            @test wigner.R_av[i] ≈ starting_R_pos[i] atol = 1e-4
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    test_dyn_scha_converged()
end

