
function init_calculator(calc, wigner_distribution :: WignerDistribution{T}, Atoms) where {T <: AbstractFloat}

    ase_positions = get_ase_positions(wigner_distribution.R_av, wigner_distribution.masses) 
    crystal = Atoms(wigner_distribution.atoms, positions = ase_positions, pbc = true, 
                       cell = wigner_distribution.cell ./ CONV_BOHR)
    crystal.calc = calc
    return crystal

end


@doc raw"""
    get_ase_positions!(ase_pos, pos , masses) 
    get_ase_positions(pos :: AbstractVector{T}, masses :: AbstractVector{T}) :: Matrix{T} where T 

Convert rescaling the sqrt of the masses in the correct ase order.
And also transform in Angstrom

This subroutine comes either with the nonallocating and allocating versions
"""
function get_ase_positions!(ase_pos :: AbstractMatrix{T}, pos :: AbstractVector{T}, masses :: AbstractVector{U})  where {T,U}
    if length(masses) != length(pos)
        error("masses and positions have different lengths")
    end

    N_atoms = length(masses)÷3
    for i in 1:N_atoms
        for j in 1:3
            index = (i-1)*3 + j
            ase_pos[i, j] = pos[index] / sqrt(masses[index]) / CONV_BOHR
        end
    end
    # new_pos = permutedims(reshape(pos ./ sqrt.(masses), 3, N_atoms)) ./ CONV_BOHR
    return new_pos
end
function get_ase_positions(pos :: AbstractVector{T}, masses :: AbstractVector{U}) :: Matrix{T} where {T,U} 
    N_atoms = length(masses)÷3
    ret = zeros(T, N_atoms, 3)

    get_ase_positions!(ret, pos, masses)
    return ret
end


function get_ase_positions_array(pos , masses) 
    if length(masses) != length(pos)
        error("masses and positions have different lengths")
    end
    N_atoms = Int64(length(masses)/3.0)
    new_pos = pos ./ sqrt.(masses) ./ CONV_BOHR
    return new_pos
end
    

@doc raw"""
    compute_configuration!(forces, stress, calculator :: PyObject, positions, masses) :: Float64
    compute_configuration!(forces, stress, calculator :: Function, positions :: AbstractVector{T}, masses) :: T


Use a given Calculator (even an ASE object) to compute energies forces and stress of a specific 
configuration.
Alternatively, it can be a function that takes atomic coordinates 
in the form of a vector n_dims * n_atoms (n_dims fast index)
in Ha atomic units, compute the forces, energies and stress tensor in atomic units.

Notably, the positions are a 1D array of mass-rescaled atomic coordinates as stored inside the ensemble.

It is possible to use a cache array to store the temporary variable for the atomic coordinates.
In the case of function, it must be the same size as position, otherwise of the correct size for the ASE object (nat, 3)

if the number of dimension is different from 3, it must be explicitly provided

In case of a function, it must be have the following signature:

```julia
calculator!(forces :: AbstractVector{T}, stress :: AbstractVector{T}, positions :: AbstractVector{T}) :: T 
```

where forces and stress are in-place modified, while the energy is returned
"""
function compute_configuration!(forces :: AbstractVector{T}, stress :: AbstractVector{T}, calculator :: PyObject, positions :: AbstractVector{T}, masses; cache=nothing, n_dims = 3) :: T where T
    nat = length(forces) ÷ 3
    if cache == nothing
        cache = zeros(T, 3, nat)
    end

    @assert n_dims == 3 "Only 3D is supported for an ASE calculator"

    get_ase_positions!(cache, positions, masses)
    calculator.positions = cache
    energy = calculator.get_potential_energy() * CONV_RY
    forces .= reshape(calculator.get_forces(), :)
    forces ./= sqrt.(masses)
    forces .*= CONV_RY / CONV_BOHR
    stress .= calculator.get_stress() * CONV_RY / CONV_BOHR^3
    

    return energy
end
function compute_configuration!(forces :: AbstractVector{T}, stress :: AbstractVector{T}, calculator :: Function, positions :: AbstractVector{T}, masses; cache=nothing, n_dims = 3) :: T where T
    nat = length(forces) ÷ n_dims

    if cache == nothing
        cache = zeros(T, n_dims*nat)
    end

    cache .= positions
    @simd for i in 1:length(cache)
        cache[i] /= sqrt(masses[i])
    end

    return calculator(forces, stress, cache)
end

