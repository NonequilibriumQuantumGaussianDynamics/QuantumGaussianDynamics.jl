__precompile__(false)
module QuanumGaussianDynamics

using LinearAlgebra
using Random
using Roots
using MPI

using Optim
using OptimizationOptimJL
using ForwardDiff

using PyCall

# Init MPI
include("parallel.jl")

"""
Some guide on the units

We adopt Hartree atomic units, where the energy is expressed in mHa (convenient for phonons).
Therefore each derived units (like forces and times) are appropriately rescaled.
"""

const SMALL_VALUE :: Float64 = 1e-8
const THR_ACOUSTIC :: Float64 = 1e-1
const CONV_FS :: Float64 = 0.048377687 # to femtoseconds
const CONV_RY :: Float64 = 0.0734985857 # eV to Ry
const CONV_BOHR :: Float64 = 1.8897261246257702 # Angstrom to Bohr
const CONV_MASS :: Float64 = 911.444175 # amu to kg
const CONV_EFIELD :: Float64 = 2.7502067*1e-7 #kVcm to E_Ry
const CONV_FREQ :: Float64 = 4.83776857*1e-5 #THz to w_Ry

export SMALL_VALUE
export CONV_FS
export CONV_RY
export CONV_BOHR
export CONV_EFIELD

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
"""
Base.@kwdef mutable struct ElectricField{T <: AbstractFloat} 
    fun :: Function #Time in fs, unit 
    Zeff :: Matrix{T} 
    edir :: Vector{T} #Must have unit norm
    eps :: Matrix{T}
end

"""
    Dynamics(dt:: Quantity,
        total_time:: Quantity,
        N :: Int;
        algorithm:: String = "generalized-verlet",
        kong_liu_ratio:: AbstractFloat = 1.0,
        verbose:: Bool = true,
        evolve_correlators:: Bool = true,
        seed:: Int = 0,
        evolve_correlated:: Bool = true,
        settings:: GeneralSettings = ASR(),
        save_filename:: String = "dynamics",
        save_correlators:: Bool = false,
        save_each:: Int=100)


The settings for the simulation. dt and total_time are in femtoseconds, or generic
time units if Unitful is used.

- `dt` is the time step [either in femtoseconds or generic time units]
- `total_time` is the total simulation time [either in femtoseconds or generic time units]
- `N` is the number of atoms in the system
- `algorithm` is the integration algorithm to use. Either "generalized-verlet" or "semi-implicit-verlet"
- `kong_liu_ratio` is the ratio exploits the importance sampling.
- `verbose` is a flag to print out information during the simulation
- `evolve_correlators` is a flag to evolve the correlators. If false, neglects bubble self-energy.
- `seed` is the seed for the random number generator
- `evolve_correlated` is a flag to extract correlated ensembles between steps. This improves the convergence of the ensemble.
- `settings` is a structure with general settings about the constrains on some modes.
- `save_filename` is the name of the file where to save the data
- `save_correlators` is a flag to save the correlators information
- `save_each` is the number of steps between each save of the data
"""
Base.@kwdef struct Dynamics{T <: AbstractFloat}
    dt :: T   #In femtosecodns
    total_time :: T # In femtosecodns
    algorithm :: String
    kong_liu_ratio :: T
    verbose :: Bool
    evolve_correlators :: Bool
    seed :: Int64
    N :: Int64
    correlated :: Bool

    # Save the data each
    save_filename :: String 
    save_correlators :: Bool 
    save_each :: Int64
end

Base.@kwdef struct GeneralSettings{T <: AbstractFloat}
    ciao :: T
    evolve_correlators :: Bool
end

"""
The WignerDistribution.

It can be initialized either via a cellconstructor Phonon object using the
method ``init_from_dyn``. Alternatively, one can initialize a generic one with the field

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
Base.@kwdef mutable struct WignerDistribution{T<: AbstractFloat}
    R_av    :: Vector{T}
    P_av    :: Vector{T}
    masses  :: Vector{T}
    n_atoms :: Int32
    n_modes :: Int32
    #RR_corr :: Symmetric{T, Matrix{T}}
    #PP_corr :: Symmetric{T, Matrix{T}}
    RR_corr :: Matrix{T}
    PP_corr :: Matrix{T}
    RP_corr :: Matrix{T}
    #alpha   :: Symmetric{T, Matrix{T}}
    #beta    :: Symmetric{T, Matrix{T}}
    alpha :: Matrix{T}
    beta :: Matrix{T}
    gamma   :: Matrix{T}

    # Eigenvalues and eigenvectors of the current Y matrix 
    λs :: Vector{T}
    λs_vect :: Matrix{T}
    evolve_correlators :: Bool
    cell :: Matrix{T}
    atoms :: Vector{String}
end

Base.@kwdef mutable struct Ensemble{T <: AbstractFloat}
    rho0 :: WignerDistribution{T}

    # stocastic displacements and forces
    # index i, j means configuration j, coordinate i
    positions :: Matrix{T}  # Positions are multiplied by the squareroot of the masses
    energies :: Vector{T} 
    forces :: Matrix{T} # Forces are divided by the squareroot of masses
    stress :: Matrix{T} # Stress in eV/A^3
    sscha_energies :: Vector{T} 
    sscha_forces :: Matrix{T}
    n_configs :: Int32
    weights :: Vector{T}
    temperature :: T
    correlated :: Bool
    y0 :: Matrix{T}

    #unit_cell :: Matrix{T}
end 

"""
Remove acoustic sum rule from eigenvalue and eigenvectors
"""
function remove_translations(vectors, values, thr)
    not_trans_mask =  values .> thr

    if sum(.!not_trans_mask) != 3
        println("WARNING")
        println("the expected number of acustic modes is 3
                #       got $(sum(.!not_trans_mask)) instead")
        println(values[:3])
    end
    #@assert sum(.!not_trans_mask) == 3   """
#Error, the expected number of acustic modes is 3
#       got $(sum(.!not_trans_mask)) instead.
#"""

    #new_values = values[ not_trans_mask ]
    #new_vectors = vectors[:, not_trans_mask]
    new_values = values[4:end]
    new_vectors = vectors[:,4:end]

    return new_vectors, new_values
end


"""
Impose the ASE projecting out the translations
"""
function apply_ASR!(matrix :: Matrix{T}, masses :: Vector{T}) where {T <: AbstractFloat}
    mass_array = zeros(T, size(matrix, 1))
end

function init_from_dyn(dyn, TEMPERATURE :: T, settings :: Dynamics{T}) where {T <: AbstractFloat}

    # Initialize the WignerDistribution structure starting from a dynamical matrix
    
    super_struct = dyn.structure.generate_supercell(dyn.GetSupercell())
    N_modes = super_struct.N_atoms * 3
    N_atoms = Int32(super_struct.N_atoms)

    w, pols = dyn.DiagonalizeSupercell() #frequencies are in Ry
    
    alpha, beta = QuanumGaussianDynamics.get_alphabeta(Float64(TEMPERATURE), w, pols)
    RR_corr, PP_corr = QuanumGaussianDynamics.get_correlators(Float64(TEMPERATURE), w, pols)
    gamma = zeros(N_modes, N_modes) #already rescaled (tilde)
    RP_corr = zeros(N_modes, N_modes) #already rescaled (tilde)
    R_av = super_struct.coords * CONV_BOHR #units
    P_av = zeros(N_atoms, 3)

    # Reshape
    R_av = reshape(permutedims(R_av), N_modes)
    P_av = reshape(permutedims(P_av), N_modes)

    # Rescale
    masses = super_struct.get_masses_array() # already in Rydberg units
    mass_array = reshape(repeat(masses',3,1), N_modes)
    R_av = R_av.*sqrt.(mass_array)
    P_av = P_av./sqrt.(mass_array)

    # Diagonalize alpha
    if settings.evolve_correlators == false
        lambda_eigen = eigen(alpha)
        λvects, λs = QuanumGaussianDynamics.remove_translations(lambda_eigen.vectors, lambda_eigen.values, THR_ACOUSTIC) #NO NEEDED WITH ALPHAS
    else
        lambda_eigen = eigen(RR_corr)
        #println("RR_Coror")
        #display(RR_corr)
        λvects, λs = QuanumGaussianDynamics.remove_translations(lambda_eigen.vectors, lambda_eigen.values, THR_ACOUSTIC) #NO NEEDED WITH ALPHAS       
    end

    # Cell
    cell = super_struct.unit_cell .*CONV_BOHR
    atoms = super_struct.atoms

    # Initialize
    rho = QuanumGaussianDynamics.WignerDistribution(R_av  = R_av, P_av = P_av, n_atoms = N_atoms, masses = mass_array, n_modes = Int32(N_modes), 
                                                alpha = alpha, beta = beta, gamma = gamma, RR_corr = RR_corr, PP_corr = PP_corr, RP_corr = RP_corr, 
                                                λs_vect = λvects, λs = λs, evolve_correlators = settings.evolve_correlators, cell = cell, atoms = atoms)
    return rho
end

include("time_evolution.jl")
include("ensemble.jl")
include("phonons.jl")
include("calculator.jl")
include("dynamics.jl")
include("external_f.jl")

end # module QuanumGaussianDynamics
