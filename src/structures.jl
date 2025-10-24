"""
Settings for the dynamics.

Dynamics(dt :: T = 1.0  #femtosecodns,
    total_time :: T # In femtosecodns,
    algorithm :: String = "generalized-verlet",
    kong_liu_ratio :: T = 1.0,
    verbose :: Bool = true,
    evolve_correlators :: Bool = true,
    correlated :: Bool = true,
    seed :: Int64,
    N :: Int64,

    # Save the data each
    save_each :: Int64 = 1,
    save_filename :: String,
    save_correlators :: Bool)


The settings for the simulation. dt and total_time are in femtoseconds

- `dt` is the time step [in femtoseconds]
- `total_time` is the total simulation time [in femtoseconds]
- `algorithm` is the integration algorithm to use. Either "generalized-verlet" or "semi-implicit-verlet"
- `kong_liu_ratio` is the ratio exploits the importance sampling.
- `verbose` is a flag to print out information during the simulation
- `evolve_correlators` is a flag to evolve the correlators <RR>, <PP> and <RP>
- `correlated` if true, the correlated approach for computing ensemble averages is used
- `seed` is the seed for the random number generator
- `N` is the number of stochastic configurations
- `save_filename` is the name of the file where to save the data
- `save_correlators` is a flag to save the correlators information
- `save_each` is the number of steps between each save of the data
"""
Base.@kwdef struct Dynamics{T<:AbstractFloat}
    dt::T = 1.0  #femtosecodns
    total_time::T # In femtosecodns
    algorithm::String = "generalized-verlet"
    kong_liu_ratio::T = 1.0
    verbose::Bool = true
    evolve_correlators::Bool = true
    correlated::Bool = true
    seed::Int64
    N::Int64

    # Save the data each
    save_each::Int64 = 1
    save_filename::String
    save_correlators::Bool
end

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
Ensemble average information
stocastic displacements and forces have index i, j, coordinate i, configuration j

Ensemble(rho0 :: WignerDistribution{T},
    positions :: Matrix{T},  # Positions are multiplied by the square root of the masses
    energies :: Vector{T},
    forces :: Matrix{T}, # Forces are divided by the squareroot of masses
    stress :: Matrix{T}, # Stress in eV/A^3
    sscha_energies :: Vector{T},
    sscha_forces :: Matrix{T},
    n_configs :: Int32,
    weights :: Vector{T},
    temperature :: T,
    correlated :: Bool,
    y0 :: Matrix{T},
    #unit_cell :: Matrix{T})
"""
Base.@kwdef mutable struct Ensemble{T<:AbstractFloat}
    rho0::WignerDistribution{T}

    # stocastic displacements and forces
    # index i, j means configuration j, coordinate i
    positions::Matrix{T}  # Positions are multiplied by the squareroot of the masses
    energies::Vector{T}
    forces::Matrix{T} # Forces are divided by the squareroot of masses
    stress::Matrix{T} # Stress in eV/A^3
    sscha_energies::Vector{T}
    sscha_forces::Matrix{T}
    n_configs::Int32
    weights::Vector{T}
    temperature::T
    correlated::Bool
    y0::Matrix{T}
end

"""
Electric Field.

    Base.@kwdef mutable struct ElectricField{T <: AbstractFloat} 
        fun :: Function #Time in fs, unit 
        Zeff :: Matrix{T} 
        edir :: Vector{T} #Must have unit norm
        eps :: Matrix{T}
    end

This structure contains the information about the external IR electric field.

- `fun` is the function that describes the electric field as a function of time
- `Zeff` is the effective charge matrix
- `edir` is the direction of the electric field
- `eps` is the dielectric constant matrix
"""
Base.@kwdef mutable struct ElectricField{T<:AbstractFloat}
    fun::Function #Time in fs, unit 
    Zeff::Matrix{T}
    edir::Vector{T} #Must have unit norm
    eps::Matrix{T}
end



"""
Basic constants necessary for conversions are defined
"""

const SMALL_VALUE::Float64 = 1e-8
const THR_ACOUSTIC::Float64 = 1e-1
const CONV_FS::Float64 = 0.048377687 # to femtoseconds
const CONV_RY::Float64 = 0.0734985857 # eV to Ry
const CONV_BOHR::Float64 = 1.8897261246257702 # Angstrom to Bohr
const CONV_MASS::Float64 = 911.444175 # amu to kg
const CONV_EFIELD::Float64 = 2.7502067*1e-7 #kVcm to E_Ry
const CONV_FREQ::Float64 = 4.83776857*1e-5 #THz to w_Ry

export SMALL_VALUE
export CONV_FS
export CONV_RY
export CONV_BOHR
export CONV_EFIELD
