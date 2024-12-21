import AtomicSymmetries: get_symmetry_group_from_spglib

@doc raw"""
    get_symmetry_group_from_spglib(wigner; kwargs...) :: Symmetries

Optionally, the spglib python module can be passed through

Here we extend the functionality of the `get_symmetry_group_from_spglib` from the AtomicSymmetry package to accept a `WignerDistribution` object.
This exploits the multiple dispatch capabilities of Julia to extend the functionality of the function.
"""
function get_symmetry_group_from_spglib(wigner :: WignerDistribution{T}; kwargs...) :: Symmetries where {T}
    ndims = 3
    @assert ndims == get_ndims(wigner) "Error: ndims must be 3 to use spglib"

    # Get the cell and crystal coordinates
    positions = zeros(T, 3, get_natoms(wigner))
    crystal = similar(positions)
    cell = zeros(T, 3, 3)
    types = map_to_int(wigner.atoms)

    positions .= reshape(wigner.R_av, 3, :)
    positions ./= .√(reshape(wigner.masses, 3, :))
    cell .= wigner.cell 
    types = map_to_int(wigner.atoms)

    get_crystal_coords!(crystal, positions, cell)
    cell ./= max(cell...)

    # println("Crystal: ", crystal')
    # println("Cell: ", cell')
    # println("Types: ", types)

    # Now get the spglib
    get_symmetry_group_from_spglib(crystal, cell, types; kwargs...)
end

function get_spglib_cell(wigner :: WignerDistribution{T}) :: Cell where {T <: AbstractFloat} 
    ndims = 3
    nat = get_natoms(wigner)

    @assert ndims == get_ndims(wigner) "Error: ndims must be 3 to use spglib"

    # Extract the position in cartesian coordinates
    cart_pos = zeros(T, ndims, nat)
    cart_pos .= reshape(wigner.R_av, ndims, :)
    cart_pos ./= .√(reshape(wigner.masses, ndims, :))
    crystal = zeros(T, ndims, nat)
    cell = Vector{Vector{T}}(undef, ndims)
    for i in 1:ndims
        cell[i] = wigner.cell[:, i]
    end

    # Extract the crystalline coordinates
    get_crystal_coords!(crystal, reshape(cart_pos, ndims, nat), wigner.cell)

    crystal_vect = Vector{Vector{T}}(undef, nat)
    for i in 1:nat
        crystal_vect[i] = crystal[:, i]
    end

    # Map atomic element name into unique integers
    types = map_to_int(wigner.atoms)

    Cell(wigner.cell, crystal_vect, types)
end



function map_to_int(list :: Vector{String}) :: Vector{Int}
    unique_el = Dict()
    [get!(unique_el, el, length(unique_el)) for el in list]
end
