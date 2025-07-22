
function init_calculator(calc, wigner_distribution :: WignerDistribution{T}, Atoms) where {T <: AbstractFloat}

    ase_positions = get_ase_positions(wigner_distribution.R_av, wigner_distribution.masses) 
    crystal = Atoms(wigner_distribution.atoms, positions = ase_positions, pbc = true, 
                       cell = wigner_distribution.cell ./ CONV_BOHR)
    crystal.calc = calc
    return crystal

end


function get_ase_positions(pos , masses) 
    if length(masses) != length(pos)
        error("masses and positions have different lengths")
    end
    N_atoms = Int64(length(masses)/3.0)
    new_pos = permutedims(reshape(pos ./ sqrt.(masses), 3, N_atoms)) ./ CONV_BOHR
    return new_pos
end


function get_ase_positions_array(pos , masses) 
    if length(masses) != length(pos)
        error("masses and positions have different lengths")
    end
    N_atoms = Int64(length(masses)/3.0)
    new_pos = pos ./ sqrt.(masses) ./ CONV_BOHR
    return new_pos
end
    

