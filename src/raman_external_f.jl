
@doc raw"""
    StimulatedRamanField <: ExternalPerturbation

An external perturbation to be used for (off-resonant) stimulated
Raman scattering. 
This perturbation describes the reaction of the system to two external
fields, which are used to stimulate the Raman scattering process.

This subroutine assumes that the electrif fields have an envelope
function that modulates the intensity of the field as a function of time
times a cosine function. The envelope functions are given by `field_env1`
and `field_env2`, and the cosine function is modulated by the central frequency
of the fields `ω1` and `ω2`. Note that they can be equal, in this
case the force is modulated by the product of the two fields intensity.

This can be created with the `get_impulsive_raman_pump` function.
"""
mutable struct StimulatedRamanField{T} <: ExternalPerturation
    field_env1 :: Function      
    field_env2 :: Function
    ω1 :: T
    ω2 :: T
    raman_tensor :: Array{T, 3} # E field, E field, n_dims*nat (n_dims * atom + i coords)
    field_1_pol :: AbstractVector{T}
    field_2_pol :: AbstractVector{T}
end

@doc raw"""
    get_impulsive_raman_pump(raman_tensor :: Array{T, 3}, field_frequency :: Quantity, field_duration :: Quantity, field_polarization:: AbstractVector{T}) :: StimulatedRamanField{T}

Create a StimulatedRamanField object for an impulsive Raman pump.
The pulse is assumed to be one (interacting twice with the system) and off-resonant with the electrons.

The time envelope of the field is assumed to be a Gaussian function.

# Arguments

- `raman_tensor::Array{T, 3}`: The Raman tensor of the system. Can be obtained from a Phonons object via `get_raman_tensor_from_phonons`.
- `field_frequency::Quantity`: The frequency of the field. A Unitful quantity with annotated units. (Note, this is a frequency, not a wavelength. Good units are THz or c/cm, not eV or nm)
- `field_duration::Quantity`: The duration of the field. A Unitful quantity with annotated units.
- `field_total_energy::Quantity`: The energy of the field pulse. A Unitful quantity with annotated units (i.e. mJ)
- `field_polarization::AbstractVector{T}`: The polarization of the field.
- `second_polarization::AbstractVector{T}`: The polarization of the second field. If not given, it is assumed to be the same as the first field. This is useful if the light is unpolarized or circularly polarized.


Note that, differently for IR pulses, it is required to provide the total energy of the field pulse, not the amplitude of the electric field. Also in this case, the microscopic value is considered (the one that interacts with the atoms), not the external one.
"""
function get_impulsive_raman_pump(raman_tensor :: Array{T, 3}, field_frequency :: Quantity, field_duration :: Quantity, field_start_time :: Quantity, field_total_energy :: Quantity, field_polarization:: AbstractVector{T}; second_polarization :: AbstractVector{T} = nothing) :: StimulatedRamanField{T} where T 
    field_duration = ustrip(auconvert(field_duration))
    field_start_time = ustrip(auconvert(field_start_time))
    ω = ustrip(auconvert(field_frequency)) / (2π)
    field_total_energy = ustrip(auconvert(field_intensity))

    # Normalize the polarization
    field_polarization ./= norm(field_polarization)

    if second_polarization === nothing
        field_2_pol = copy(field_polarization)
    else
        field_2_pol = second_polarization / norm(second_polarization)
    end

    function field(t)
        return sqrt(field_total_energy) * exp(-0.25 * (t - field_start_time)^2 / field_duration^2) / sqrt(sqrt(2π * field_duration^2))
    end


    StimulatedRamanField(field, field, ω, ω, raman_tensor, field_polarization, field_polarization, field_2_pol)
end

@doc raw"""
    get_raman_tensor_from_phonons(py_dyn) :: Array{T, 3}

Read the raman tensor from a cellconstructor Phonons object.
"""
function get_raman_tensor_from_phonons(py_dyn) :: Array{Float64, 3}
    @assert py_dyn.raman_tensor isa Array "Raman tensor must be a 3D array of Float64"
    ndims, _, nat3 = size(py_dyn.raman_tensor)

    raman_tensor = zeros(Float64, ndims, ndims, nat)
    for i in nat3
        @views raman_tensor[:, :, i] .= py_dyn.raman_tensor[:, :, i]
    end
    raman_tensor
end



function get_external_forces(t :: T, raman_field :: StimulatedRamanField{T},
        wigner :: WignerDistribution{T}) :: Vector{T} where T

    n_dims = get_ndims(wigner)
    nat = get_natoms(wigner)

    @assert size(raman_field.raman_tensor, 1) == n_dims "Raman tensor must have the same number of dimensions as the system. $(size(raman_field.raman_tensor, 1) != n_dims)"

    forces = zeros(T, n_dims*nat)

    tmp_vect = zeros(T, n_dims)
    phase_factor = raman_field.field_env1(t) * raman_field.field_env2(t)
    phase_factor *= cos((raman_field.ω1 - raman_field.ω2)*t)

    for i in 1:nat*n_dims
        start = n_dims*(i-1) + 1
        fin = start + n_dims - 1

        @views mul!(tmp_vect, raman_field.field_1_pol, 
                    raman_field.raman_tensor[:, :, i])
        @views mul!(forces[start:fin], tmp_vect, raman_field.field_2_pol')
    end

    forces .*= phase_factor
    forces
end
