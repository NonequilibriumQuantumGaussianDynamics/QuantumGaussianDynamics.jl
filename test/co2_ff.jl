using ReverseDiff
using Unitful, UnitfulAtomic
using LinearAlgebra

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

