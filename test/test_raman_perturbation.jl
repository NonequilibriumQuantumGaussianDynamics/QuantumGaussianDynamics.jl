using ReverseDiff
using Unitful, UnitfulAtomic
using LinearAlgebra
using DelimitedFiles
using Unitful, UnitfulAtomic
using PhysicalConstants 
using AtomicSymmetries
using Test

using QuantumGaussianDynamics

"""
Use a CH2 harmonic forcefield to test the Raman perturbation
"""

const k_oc = 1.3288
const k_θ = 0.002225
const CO_length = 2.19775 # Atomic units
const CO_angle = 2π 
const raman_c1 = 0.0243
const raman_c2 = 0.006957

function my_energy(positions)
    energy = 0
    dist1 = norm(positions[1:3] - positions[4:6])
    dist2 = norm(positions[4:6] - positions[7:9])

    v1 = positions[1:3] - positions[4:6]
    v2 = positions[7:9] - positions[4:6]
    division = dot(v1, v2)/(dist1*dist2) * 0.999999999 # Regularize to avoid diff acos(1.0) = NaN
    θ = acos(division)

    energy += 0.5 * k_oc * (dist1 - CO_length)^2
    energy += 0.5 * k_oc * (dist2 - CO_length)^2
    energy += 0.5 * k_θ * (θ - CO_angle)^2
    energy
end

const co2_energy_tape = ReverseDiff.GradientTape(my_energy, rand(9))
const c_co2_energy_tape = ReverseDiff.compile(co2_energy_tape)

function co2_force_field!(forces, stress, positions)
    stress .= 0.0
    results = DiffResults.GradientResult(positions)
    ReverseDiff.gradient!(results, c_co2_energy_tape, positions)
    forces .= -DiffResults.gradient(results)
    DiffResults.value(results)
end

function py_force_field(positions)

    forces = zeros(Float64, 9)
    stress = zeros(Float64, 9, 9)
    my_positions = zeros(Float64, 9)
    annotated_pos = positions .* u"Å"
    my_positions .= ustrip.(auconvert.(annotated_pos))

    energy = co2_force_field!(forces, stress, my_positions)

    forces = ustrip.(uconvert.(u"eV/Å", forces * aunit(u"eV/Å")))
    return ustrip(uconvert(u"eV", energy * aunit(u"eV"))), forces
end

function create_co2()
    positions = zeros(Float64, 9)
    positions[1] = -CO_length
    positions[7] = CO_length

    raman_tensor = zeros(Float64, 3, 3, 9)
    raman_tensor[1, 1, 1] = -raman_c1
    raman_tensor[2, 2, 1] = -raman_c2
    raman_tensor[3, 3, 1] = -raman_c2
    raman_tensor[2, 1, 2] = -raman_c2
    raman_tensor[1, 2, 2] = -raman_c2
    raman_tensor[3, 1, 3] = -raman_c2
    raman_tensor[1, 3, 3] = -raman_c2

    raman_tensor[:, :, 7:9] .= -raman_tensor[:, :, 1:3]

    return positions, raman_tensor
end


function test_cleaning(; verbose=false)
    algorithm = "generalized-verlet"
    dt = 0.2u"fs"
    total_time = 100.0u"fs"
    N_configs = 100
    n_atoms = 3
    temperature = 0.0u"K"

    raman_polarization = [1.0, 0.0, 0.0]


    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                settings = ASRfixmodes(; small_w_value = 1e-5),
                                                save_each=1,
                                                save_filename="co2_nothing")
    # Impose the cleaning of gradients
    set_clean_gradients!(settings, true)
    wigner = WignerDistribution(n_atoms; n_modes = 3n_atoms - 5)

    # Prepare the initial conditions
    pos, raman_tensor = create_co2()
    wigner.R_av .= pos

    mass = [14582.2, 14582.2, 14582.2, 10947.4, 10947.4, 10947.4, 14582.2, 14582.2, 14582.2]
    mass .*= 2
    wigner.masses .= mass
    wigner.R_av .*= sqrt.(mass)

    # Diagonalize the dynamical matrix to get the RR and PP correlators
    phi = readdlm(joinpath(@__DIR__, "co2phi.dat"))
    dynmat = phi ./ sqrt.(mass * mass')
    eigenvals, eigenvecs = eigen(dynmat)

    ω = sqrt.(abs.(eigenvals))
    # Remove the first 5 modes (rotations)
    ω[1:5] .= 0.0

    println("frequencies: $(ustrip.(uconvert.(u"c/cm", ω * aunit(u"eV")/(2π * PhysicalConstants.CODATA2018.ħ)))) cm-1")

    # Generate the appropriate initial conditions
    RR, PP = QuantumGaussianDynamics.get_correlators(temperature, ω, eigenvecs, get_general_settings(settings))
    wigner.RR_corr .= RR
    wigner.PP_corr .= PP

    QuantumGaussianDynamics.update!(wigner, settings)

    # Define the ensemble
    ensemble = QuantumGaussianDynamics.Ensemble(wigner, settings; n_configs=N_configs, temperature=temperature)
    if verbose
        @show raman_tensor
    end
    raman_field = get_impulsive_raman_pump(raman_tensor, 1.0/550u"nm/c", 10.0u"fs", 15.0u"fs", 0.0u"GV/mm", 
                                           raman_polarization)

    symmetry_group = get_symmetry_group_from_spglib(wigner)
    n_sym_before = length(symmetry_group.symmetries)
    if verbose
        println("Number of symmetries: ", length(symmetry_group.symmetries))
    end

    # Filter the invariant symmetries with respect to the Raman perturbation
    perturbation_direction = get_perturbation_direction(raman_field, wigner)
    if verbose
        println("Perturbation direction: ", perturbation_direction)
    end
    @test norm(perturbation_direction) ≈ 1.0
    @test abs(perturbation_direction[1]) > 0.1
    @test perturbation_direction[7] ≈ -perturbation_direction[1]

    filter_invariant_symmetries!(symmetry_group, perturbation_direction)

    # The raman perturbation should not change the number of symmetries
    @test length(symmetry_group.symmetries) == n_sym_before

    if verbose
        println("Number of symmetries after Raman activity: ", length(symmetry_group.symmetries))
    end

    QuantumGaussianDynamics.generate_ensemble!(N_configs, ensemble, wigner)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, co2_force_field!)

    # Get the external force
    ext_for = QuantumGaussianDynamics.get_external_forces(0.0, raman_field, wigner)
    if verbose
        @show ext_for
    end

    rr_copy = copy(wigner.RR_corr)
    r_copy = copy(wigner.R_av)

    # Check that everything runs without errors
    QuantumGaussianDynamics.integrate!(wigner, ensemble, settings, co2_force_field!, raman_field)

    # Check that the correlators are not changed
    for i in 1:length(wigner.RR_corr)
        @test wigner.RR_corr[i] ≈ rr_copy[i]
    end
    for i in 1:length(wigner.R_av)
        @test wigner.R_av[i] ≈ r_copy[i]
    end
end


function test_raman_perturbation(; verbose=false)
    algorithm = "generalized-verlet"
    dt = 0.2u"fs"
    total_time = 100.0u"fs"
    N_configs = 100
    n_atoms = 3
    temperature = 0.0u"K"

    raman_polarization = [1.0, 0.0, 0.0]


    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                settings = ASRfixmodes(; small_w_value = 1e-5),
                                                save_each=1,
                                                save_filename="co2_raman")
    # Impose the cleaning of gradients
    set_clean_gradients!(settings, true)

    # TODO: The set clean gradients does not work as expected

    wigner = WignerDistribution(n_atoms; n_modes = 3n_atoms - 5)

    # Prepare the initial conditions
    pos, raman_tensor = create_co2()
    wigner.R_av .= pos

    mass = [14582.2, 14582.2, 14582.2, 10947.4, 10947.4, 10947.4, 14582.2, 14582.2, 14582.2]
    mass .*= 2
    wigner.masses .= mass
    wigner.R_av .*= sqrt.(mass)

    # Diagonalize the dynamical matrix to get the RR and PP correlators
    phi = readdlm(joinpath(@__DIR__, "co2phi.dat"))
    dynmat = phi ./ sqrt.(mass * mass')
    eigenvals, eigenvecs = eigen(dynmat)

    ω = sqrt.(abs.(eigenvals))
    # Remove the first 5 modes (rotations)
    ω[1:5] .= 0.0

    println("frequencies: $(ustrip.(uconvert.(u"c/cm", ω * aunit(u"eV")/(2π * PhysicalConstants.CODATA2018.ħ)))) cm-1")

    # Generate the appropriate initial conditions
    RR, PP = QuantumGaussianDynamics.get_correlators(temperature, ω, eigenvecs, get_general_settings(settings))
    wigner.RR_corr .= RR
    wigner.PP_corr .= PP

    QuantumGaussianDynamics.update!(wigner, settings)

    # Define the ensemble
    ensemble = QuantumGaussianDynamics.Ensemble(wigner, settings; n_configs=N_configs, temperature=temperature)
    if verbose
        @show raman_tensor
    end
    raman_field = get_impulsive_raman_pump(raman_tensor, 1.0/550u"nm/c", 10.0u"fs", 15.0u"fs", 1.0e2u"GV/mm", 
                                           raman_polarization)

    symmetry_group = get_symmetry_group_from_spglib(wigner)
    n_sym_before = length(symmetry_group.symmetries)
    if verbose
        println("Number of symmetries: ", length(symmetry_group.symmetries))
    end

    # Filter the invariant symmetries with respect to the Raman perturbation
    perturbation_direction = get_perturbation_direction(raman_field, wigner)
    if verbose
        println("Perturbation direction: ", perturbation_direction)
    end
    @test norm(perturbation_direction) ≈ 1.0
    @test abs(perturbation_direction[1]) > 0.1
    @test perturbation_direction[7] ≈ -perturbation_direction[1]

    filter_invariant_symmetries!(symmetry_group, perturbation_direction)

    # The raman perturbation should not change the number of symmetries
    @test length(symmetry_group.symmetries) == n_sym_before

    if verbose
        println("Number of symmetries after Raman activity: ", length(symmetry_group.symmetries))
    end

    QuantumGaussianDynamics.generate_ensemble!(N_configs, ensemble, wigner)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, co2_force_field!)

    # Get the external force
    ext_for = QuantumGaussianDynamics.get_external_forces(0.0, raman_field, wigner)
    if verbose
        @show ext_for
    end

    # Check that everything runs without errors
    QuantumGaussianDynamics.integrate!(wigner, ensemble, settings, co2_force_field!, raman_field)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_raman_perturbation(; verbose=true)
end
