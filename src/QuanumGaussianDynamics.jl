module QuanumGaussianDynamics

using LinearAlgebra

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
end


struct Ensemble{T <: AbstractFloat}
    original_RR_corr :: Matrix{T} # All quantities with the tilde
    original_R_av :: Matrix # All these quantities are with the tilde
    λs_inv :: Vector{T}
    positions :: Matrix{T}  # Positions are multiplied by the squareroot of the masses
    n_configs :: Int32
    weights :: Vector{T}
end 

include("time_evolution.jl")
include("ensemble.jl")

end # module QuanumGaussianDynamics
