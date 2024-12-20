using ReverseDiff
using Unitful, UnitfulAtomic
using LinearAlgebra
using DelimitedFiles
using Unitful, UnitfulAtomic
using PhysicalConstants 
using AtomicSymmetries

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

function test_raman_perturbation()
    algorithm = "generalized-verlet"
    dt = 0.2u"fs"
    total_time = 20.0u"fs"
    N_configs = 100
    n_atoms = 3
    temperature = 0.0u"K"

    raman_polarization = [1.0, 0.0, 0.0]

    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                settings = ASR(; ignore_small_w=true),
                                                save_each=1,
                                                save_filename="co2_raman")


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
    @show raman_tensor
    raman_field = get_impulsive_raman_pump(raman_tensor, 1.0/550u"nm/c", 10.0u"fs", 15.0u"fs", 1.0u"MV/cm", 
                                           raman_polarization)

    symmetry_group = get_symmetry_group_from_spglib(wigner)
    println("Number of symmetries: ", length(symmetry_group.symmetries))

    # Filter the invariant symmetries with respect to the Raman perturbation
    filter_invariant_symmetries!(symmetry_group, get_perturbation_direction(raman_field, wigner))
    println("Number of symmetries after Raman activity: ", length(symmetry_group.symmetries))

    QuantumGaussianDynamics.generate_ensemble!(N_configs, ensemble, wigner)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, co2_force_field!)

    # Get the external force
    ext_for = QuantumGaussianDynamics.get_external_forces(0.0, raman_field, wigner)
    @show ext_for

    # Use the e


    QuantumGaussianDynamics.integrate!(wigner, ensemble, settings, co2_force_field!, raman_field)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_raman_perturbation()
end
