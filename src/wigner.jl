"""
The WignerDistribution.

It can be initialized using the method init_from_dyn. Alternatively, one can initialize a generic one with the field

    WignerDistribution(n_atoms; type = Float64, n_dims=3)

The structure contains the following fields:

    Base.@kwdef mutable struct WignerDistribution{T<: AbstractFloat}
        R_av    :: Vector{T}
        P_av    :: Vector{T}
        masses  :: Vector{T}
        n_atoms :: Int
        n_modes :: Int
        RR_corr :: Matrix{T}
        PP_corr :: Matrix{T}
        RP_corr :: Matrix{T}
        alpha :: Matrix{T}
        beta :: Matrix{T}
        gamma   :: Matrix{T}
        
        settings :: GeneralSettings
        λs :: Vector{T}
        λs_vect :: Matrix{T}
        evolve_correlators :: Bool
        cell :: Matrix{T}
        atoms :: Vector{String}
    end



Note that all the variable here are with a tilde (mass rescaled)
So that we can use linear-algebra on them quickly.
""" 

Base.@kwdef mutable struct WignerDistribution{T<:AbstractFloat}
    R_av::Vector{T}
    P_av::Vector{T}
    masses::Vector{T}
    n_atoms::Int32
    n_modes::Int32
    RR_corr::Matrix{T}
    PP_corr::Matrix{T}
    RP_corr::Matrix{T}
    alpha::Matrix{T}
    beta::Matrix{T}
    gamma::Matrix{T}

    # Eigenvalues and eigenvectors of the current Y matrix
    λs::Vector{T}
    λs_vect::Matrix{T}
    evolve_correlators::Bool
    cell::Matrix{T}
    atoms::Vector{String}
end

"""
    init_from_dyn(dyn, TEMPERATURE::T, settings::Dynamics{T}) where {T<:AbstractFloat}

Initialize a `WignerDistribution` object starting from a dynamical matrix.

This routine constructs the equilibrium nuclear quantum state (in terms of
positions, momenta, and correlation matrices) associated with the phonon
modes of the provided dynamical matrix at a given temperature.

# Arguments
- `dyn`: A dynamical matrix object from SSCHA calculations
- `TEMPERATURE::T`: The target temperature.
- `settings::Dynamics{T}`: Simulation settings.

# Details
- Builds a supercell from `dyn` and computes the number of modes and atoms.
- Diagonalizes the supercell dynamical matrix to obtain phonon frequencies and
  eigenvectors (polarizations).
- Computes Gaussian width parameters (`alpha`, `beta`) and correlators
  (`RR_corr`, `PP_corr`, `RP_corr`) at the specified temperature.
- Initializes average positions (`R_av`) and momenta (`P_av`), rescaled by
  atomic masses.
- Removes translational acoustic modes from the correlators/eigenvalues.

# Returns
- `rho::WignerDistribution`: An initialized Wigner distribution representing the
  quantum nuclear state corresponding to the given dynamical matrix and
  temperature.

# Notes
- Frequencies are assumed to be in Rydberg units.
- Cell vectors and coordinates are converted to Bohr units.
- If `settings.evolve_correlators == false`, eigen-decomposition of `alpha`
  is used; otherwise, the `RR_corr` matrix is diagonalized (deprecated).
"""
function init_from_dyn(dyn, TEMPERATURE::T, settings::Dynamics{T}) where {T<:AbstractFloat}

    # Initialize the WignerDistribution structure starting from a dynamical matrix

    super_struct = dyn.structure.generate_supercell(dyn.GetSupercell())
    N_modes = super_struct.N_atoms * 3
    N_atoms = Int32(super_struct.N_atoms)

    w, pols = dyn.DiagonalizeSupercell() #frequencies are in Ry

    alpha, beta = QuantumGaussianDynamics.get_alphabeta(T(TEMPERATURE), w, pols)
    RR_corr, PP_corr = QuantumGaussianDynamics.get_correlators(T(TEMPERATURE), w, pols)
    gamma = zeros(N_modes, N_modes) #already rescaled (tilde)
    RP_corr = zeros(N_modes, N_modes) #already rescaled (tilde)
    R_av = super_struct.coords * CONV_BOHR #units
    P_av = zeros(N_atoms, 3)

    # Reshape
    R_av = reshape(permutedims(R_av), N_modes)
    P_av = reshape(permutedims(P_av), N_modes)

    # Rescale
    masses = super_struct.get_masses_array() # already in Rydberg units
    mass_array = reshape(repeat(masses', 3, 1), N_modes)
    R_av .*= sqrt.(mass_array)
    P_av ./= sqrt.(mass_array)

    # Diagonalize alpha
    if settings.evolve_correlators == false
        lambda_eigen = eigen(alpha)
        λvects, λs = QuantumGaussianDynamics.remove_translations(
            lambda_eigen.vectors,
            lambda_eigen.values,
            THR_ACOUSTIC,
        ) #NO NEEDED WITH ALPHAS
    else
        lambda_eigen = eigen(RR_corr)
        λvects, λs = QuantumGaussianDynamics.remove_translations(
            lambda_eigen.vectors,
            lambda_eigen.values,
            THR_ACOUSTIC,
        ) #NO NEEDED WITH ALPHAS
    end

    # Cell
    cell = super_struct.unit_cell .* CONV_BOHR
    atoms = super_struct.atoms

    # Initialize
    rho = QuantumGaussianDynamics.WignerDistribution(
        R_av = R_av,
        P_av = P_av,
        n_atoms = N_atoms,
        masses = mass_array,
        n_modes = Int32(N_modes),
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        RR_corr = RR_corr,
        PP_corr = PP_corr,
        RP_corr = RP_corr,
        λs_vect = λvects,
        λs = λs,
        evolve_correlators = settings.evolve_correlators,
        cell = cell,
        atoms = atoms,
    )
    return rho
end

"""
Remove acoustic sum rule from eigenvalue and eigenvectors
"""
function remove_translations(
    vectors::AbstractMatrix{T},
    values::AbstractVector{T},
    thr::T,
) where {T<:AbstractFloat}
    mask = values .> thr
    nremoved = count(!, mask)
    if nremoved != 3
        @warn "Expected 3 acoustic modes, found $nremoved"
    end
    return vectors[:, mask], values[mask]
end

