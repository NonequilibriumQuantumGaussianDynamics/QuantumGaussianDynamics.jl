module QuanumGaussianDynamics

using LinearAlgebra

"""
Some guide on the units

We adopt Hartree atomic units, where the energy is expressed in mHa (convenient for phonons).
Therefore each derived units (like forces and times) are appropriately rescaled.
"""

const SMALL_VALUE :: Float64 = 1e-8
const CONV_FS :: Float64 = 0.04134137333518211  # femtoseconds
const CONV_EV :: Float64 = 3.674932217565499e-5
const CONV_ANGSTROM :: Float64 = 1.8897261246257702

"""
Here information about the dynamics are stored, 
like integration algorithm, time_step, kong_liu ratio 
and so on, so forth. 
"""
struct Dynamics{T <: AbstractFloat}
    dt :: T   #In femtosecodns
    total_time :: T # In femtosecodns
    algorithm :: String
    kong_liu_ratio :: T
    verbose :: Bool

    # Save the data each
    save_filename :: String 
    save_correlators :: Bool 
    save_each :: Int32 
end

"""
Info about the wigner distribution

Note that all the variable here are with a tilde (mass rescaled)
So that we can use linear-algebra on them quickly.
"""
struct WignerDistribution{T<: AbstractFloat}
    R_av    :: Vector{T}
    P_av    :: Vector{T}
    masses  :: Vector{T}
    n_atoms :: Int32
    RR_corr :: Matrix{T}
    PP_corr :: Matrix{T}
    RP_corr :: Matrix{T}

    # Eigenvalues and eigenvectors of the current Y matrix 
    λs :: Vector{T}
    λs_vect :: Matrix{T}
end


struct Ensemble{T <: AbstractFloat}
    original_RR_corr :: Matrix{T} # All quantities with the tilde
    original_R_av :: Matrix # All these quantities are with the tilde

    # Eigenvalues and eigenvectors of the Y matrix that generated the ensemble
    λs_inv :: Vector{T}
    λs_vect :: Matrix{T}

    positions :: Matrix{T}  # Positions are multiplied by the squareroot of the masses
    forces :: Matrix{T} # Forces are divided by the squareroot of masses
    # index i, j means configuration j, coordinate i
    n_configs :: Int32
    weights :: Vector{T}

    masses :: Vector{T}
    unit_cell :: Matrix{T}
end 

"""
Remove acoustic sum rule from eigenvalue and eigenvectors
"""
function remove_translations(vectors, values)
    not_trans_mask =  values .> SMALL_VALUE

    @assert sum(not_trans_mask) == 3   """
Error, the expected number of acustic modes is 3
       got $(sum(not_trans_mask)) instead.
"""

    new_values = values[ not_trans_mask ]
    new_vectors = vectors[:, not_trans_mask]

    return new_vectors, new_values
end


"""
Impose the ASE projecting out the translations
"""
function apply_ASR!(matrix :: Matrix{T}, masses :: Vector{T}) where {T <: AbstractFloat}
    mass_array = zeros(T, size(matrix, 1))
end

include("time_evolution.jl")
include("ensemble.jl")
include("dynamics.jl")

end # module QuanumGaussianDynamics
