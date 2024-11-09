using QuantumGaussianDynamics
using PyCall
using Unitful, UnitfulAtomic
using Test

const ω1 = 500.0u"c/cm" * 2π
const ω2 = 700.0u"c/cm" * 2π
const ω3 = 900.0u"c/cm" * 2π
const mass = 918.0u"me" 

function harmonic_calculator!(forces, stress, positions)
    energy = 0 
    stress .= 0.0
    ω = [ω1, ω2, ω3]
    k = ustrip.(auconvert.(mass * ω.^2))
    @simd for i in 1:length(positions)
        forces[i] = -k[i] * positions[i]
        energy += 0.5 * k[i] * positions[i]^2
    end
    return energy
end
function harmonic_calculator_asr!(forces, stress, positions)
    energy = 0 
    stress .= 0.0
    ω = [ω1, ω2, ω3]
    k = ustrip.(auconvert.(mass * ω.^2))

    δcoords = positions[1:3] .- positions[4:end]

    @simd for i in 1:3
        δ = positions[i] - positions[i+3]
        forces[i] = -k[i] * δ
        forces[i+3] = -forces[i]
        energy += 0.5 * k[i] * δ^2
    end
    return energy
end

@pyimport ase


function test_compute_force_ase()
    # Import a local python module
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    ase_calc_module = pyimport("test_ase_calculator")

    # Convert the force constant in ASE units
    k1 = uconvert(u"eV/Å^2", ω1^2 * mass)
    k2 = uconvert(u"eV/Å^2", ω2^2 * mass)
    k3 = uconvert(u"eV/Å^2", ω3^2 * mass)


    py_calc = ase_calc_module.Harmonic3D(ustrip(k1), 
                                         ustrip(k2), 
                                         ustrip(k3))

    wigner = WignerDistribution(1; n_dims=3)

    wigner.masses .= ustrip(auconvert(mass))
    wigner.atoms .= "H"

    calculator = QuantumGaussianDynamics.init_calculator(py_calc, wigner, ase.Atoms)

    # Get a random position
    pos = ustrip(auconvert(1.0u"Å"))
    positions = [pos, pos, pos] .* √(ustrip(auconvert(mass)))
    forces_ase = zeros(Float64, 3)
    forces_jl = similar(forces_ase)
    stress = zeros(Float64, 6)

    energy_ase = QuantumGaussianDynamics.compute_configuration!(forces_ase, stress, calculator, positions, wigner.masses)
    energy_jl = QuantumGaussianDynamics.compute_configuration!(forces_jl, stress, harmonic_calculator!, positions, wigner.masses)

    println("Energy ASE: ", energy_ase)
    println("Forces ASE: ", forces_ase)
    println("Energy JL: ", energy_jl)
    println("Forces JL: ", forces_jl)

    @test energy_ase ≈ energy_jl rtol = 1e-4
    for i in 1:3
        @test forces_ase[i] ≈ forces_jl[i] rtol = 1e-4
    end
end


function test_ase_calculator_harmonic()
    # Import a local python module
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    ase_calc_module = pyimport("test_ase_calculator")

    # Convert the force constant in ASE units
    k1 = uconvert(u"eV/Å^2", ω1^2 * mass)
    k2 = uconvert(u"eV/Å^2", ω2^2 * mass)
    k3 = uconvert(u"eV/Å^2", ω3^2 * mass)

    # Initialize the calculator
    py_calc = ase_calc_module.Harmonic3D(ustrip(k1), 
                                         ustrip(k2), 
                                         ustrip(k3))

    algorithm = "generalized-verlet"
    dt = 0.5u"fs"
    total_time = 0.1u"ps"
    N_configs = 1000


    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                settings = NoASR(),
                                                save_filename="python_ase",
                                                save_each=1)
                                               


    wigner = WignerDistribution(1; n_dims=3)

    wigner.RR_corr[1,1] = ustrip(auconvert(1/(2ω1)))
    wigner.RR_corr[2,2] = ustrip(auconvert(1/(2ω2)))
    wigner.RR_corr[3,3] = ustrip(auconvert(1/(2ω3)))
    wigner.PP_corr[1,1] = ustrip(auconvert(ω1/2))
    wigner.PP_corr[2,2] = ustrip(auconvert(ω2/2))
    wigner.PP_corr[3,3] = ustrip(auconvert(ω3/2))

    wigner.R_av .= ustrip(auconvert(0.1u"Å"))

    wigner.masses .= ustrip(auconvert(mass))
    wigner.atoms .= "H"

    calculator = QuantumGaussianDynamics.init_calculator(py_calc, wigner, ase.Atoms)


    QuantumGaussianDynamics.update!(wigner, settings)
    ensemble = QuantumGaussianDynamics.Ensemble(wigner, settings; n_configs=N_configs, temperature=0.0u"K")

    QuantumGaussianDynamics.generate_ensemble!(ensemble, wigner)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, calculator)

    efield = QuantumGaussianDynamics.fake_field(1)

    QuantumGaussianDynamics.integrate!(wigner, ensemble, settings, calculator, efield)

    @test wigner.RR_corr[1,1] ≈ ustrip(auconvert(1/(2ω1))) rtol = 1e-1
    @test wigner.RR_corr[2,2] ≈ ustrip(auconvert(1/(2ω2))) rtol = 1e-1
    @test wigner.RR_corr[3,3] ≈ ustrip(auconvert(1/(2ω3))) rtol = 1e-1

    return wigner
end

function test_julia_harmonic3d()
    # Import a local python module
    algorithm = "generalized-verlet"
    dt = 0.5u"fs"
    total_time = 0.1u"ps"
    N_configs = 1000


    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                settings = NoASR(),
                                                save_filename = "julia",
                                                save_each=1)


    wigner = WignerDistribution(1; n_dims=3)

    wigner.RR_corr[1,1] = ustrip(auconvert(1/(2ω1)))
    wigner.RR_corr[2,2] = ustrip(auconvert(1/(2ω2)))
    wigner.RR_corr[3,3] = ustrip(auconvert(1/(2ω3)))
    wigner.PP_corr[1,1] = ustrip(auconvert(ω1/2))
    wigner.PP_corr[2,2] = ustrip(auconvert(ω2/2))
    wigner.PP_corr[3,3] = ustrip(auconvert(ω3/2))

    wigner.R_av .= ustrip(auconvert(0.1u"Å"))

    wigner.masses .= ustrip(auconvert(mass))
    wigner.atoms .= "H"

    QuantumGaussianDynamics.update!(wigner, settings)
    ensemble = QuantumGaussianDynamics.Ensemble(wigner, settings; n_configs=N_configs, temperature=0.0u"K")

    QuantumGaussianDynamics.generate_ensemble!(ensemble, wigner)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, harmonic_calculator!)

    efield = QuantumGaussianDynamics.fake_field(1)

    QuantumGaussianDynamics.integrate!(wigner, ensemble, settings, harmonic_calculator!, efield)

    @test wigner.RR_corr[1,1] ≈ ustrip(auconvert(1/(2ω1))) rtol = 1e-1
    @test wigner.RR_corr[2,2] ≈ ustrip(auconvert(1/(2ω2))) rtol = 1e-1
    @test wigner.RR_corr[3,3] ≈ ustrip(auconvert(1/(2ω3))) rtol = 1e-1

    return wigner
end



function test_asr_harmonic_py()
    # Import a local python module
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    ase_calc_module = pyimport("test_ase_calculator")

    # Convert the force constant in ASE units
    k1 = uconvert(u"eV/Å^2", ω1^2 * mass)
    k2 = uconvert(u"eV/Å^2", ω2^2 * mass)
    k3 = uconvert(u"eV/Å^2", ω3^2 * mass)

    # Initialize the calculator
    py_calc = ase_calc_module.Harmonic3D_ASR(ustrip(k1), 
                                             ustrip(k2), 
                                             ustrip(k3))

    algorithm = "generalized-verlet"
    dt = 0.5u"fs"
    total_time = 0.1u"ps"
    N_configs = 1000


    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                settings = NoASR(),
                                                save_filename="python_ase",
                                                save_each=1)
                                               


    wigner = WignerDistribution(2; n_dims=3)

    wigner.RR_corr[1,1] = ustrip(auconvert(1/(2ω1)))
    wigner.RR_corr[2,2] = ustrip(auconvert(1/(2ω2)))
    wigner.RR_corr[3,3] = ustrip(auconvert(1/(2ω3)))
    wigner.PP_corr[1,1] = ustrip(auconvert(ω1/2))
    wigner.PP_corr[2,2] = ustrip(auconvert(ω2/2))
    wigner.PP_corr[3,3] = ustrip(auconvert(ω3/2))

    wigner.R_av .= ustrip(auconvert(0.1u"Å"))

    wigner.masses .= ustrip(auconvert(mass))
    wigner.atoms .= "H"

    calculator = QuantumGaussianDynamics.init_calculator(py_calc, wigner, ase.Atoms)


    QuantumGaussianDynamics.update!(wigner, settings)
    ensemble = QuantumGaussianDynamics.Ensemble(wigner, settings; n_configs=N_configs, temperature=0.0u"K")

    QuantumGaussianDynamics.generate_ensemble!(ensemble, wigner)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, calculator)

    efield = QuantumGaussianDynamics.fake_field(1)

    QuantumGaussianDynamics.integrate!(wigner, ensemble, settings, calculator, efield)

    @test wigner.RR_corr[1,1] ≈ ustrip(auconvert(1/(2ω1))) rtol = 1e-1
    @test wigner.RR_corr[2,2] ≈ ustrip(auconvert(1/(2ω2))) rtol = 1e-1
    @test wigner.RR_corr[3,3] ≈ ustrip(auconvert(1/(2ω3))) rtol = 1e-1

    return wigner
end



if abspath(PROGRAM_FILE) == @__FILE__
    test_compute_force_ase()

    wj = test_julia_harmonic3d()
    wp = test_ase_calculator_harmonic()

    for i in 1:3
        @test wj.RR_corr[i,i] ≈ wp.RR_corr[i,i] rtol = 5e-2
        @test wj.R_av[i] ≈ wp.R_av[i] rtol = 5e-2
        @test wj.P_av[i] ≈ wp.P_av[i] rtol = 5e-2
    end
end






