module QuantumGaussianDynamics

using LinearAlgebra
using Random
using PyCall
using Roots
using MPI
#using Optimization
using Optim
using OptimizationOptimJL
using ForwardDiff

using Unitful, UnitfulAtomic
using PhysicalConstants

using AtomicSymmetries
using Spglib

# Init MPI
include("parallel.jl")

"""
Some guide on the units

We adopt Ha atomic units
Note the following units are all in Hartree. 
"""

const SMALL_VALUE :: Float64 = 1e-8
const THR_ACOUSTIC :: Float64 = 1e-1
const CONV_FS :: Float64 = 0.02418884326576744 #0.048377687 # to femtoseconds
const CONV_RY :: Float64 = 0.036749322175518594 # 0.0734985857 # eV to Ry
const CONV_BOHR :: Float64 = 1.8897261246257702 # Angstrom to Bohr
const CONV_MASS :: Float64 = 1822.888486217313 # 2*911.444175 # amu to me
const CONV_EFIELD :: Float64 = 1.9446903811416696e-7# 2.7502067*1e-7 #kVcm to E_Ry
const CONV_FREQ :: Float64 = 2.418884326576744e-5# 4.83776857*1e-5 #THz to frequency in ν_Ha
const CONV_K :: Float64 = 3.1668115634438576e-6 
# NOTE: CONV_FREQ does not incorporate a 2π factor which is expicitly
# inside the file external_f to go into a ω.

export SMALL_VALUE
export CONV_FS
export CONV_RY
export CONV_BOHR
export CONV_EFIELD


# Define abstract types
abstract type ExternalPerturbation end

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
Base.@kwdef mutable struct ElectricField{T <: AbstractFloat} <: ExternalPerturbation
    fun :: Function #Time in fs, unit 
    Zeff :: Matrix{T} 
    edir :: Vector{T} #Must have unit norm
    eps :: Matrix{T}
end

@doc raw"""
    StochasticSettings()

A structure to set the stochastic settings for the calculation.
It contains a series of flag the change the behaviour of the dynamics.
The overall results should not be affected by these flags
in the limit of a large number of configurations, 
but they can lead to better convergence in some cases.

- `clean_start_gradient_centroids` is a flag to removes the gradient of the centroids based on the initial conditions. 
Note: this flag changes the dynamics from the real one if the starting point is not the equilibrium one.
- `clean_start_gradient_fcs` is a flag to removes the gradient of the force constants based on the initial conditions.
As the previous one, this flag changes the dynamics from the real one if the starting point is not the equilibrium one.
- `remove_scha_forces` is a flag to remove the Scha forces from the calculation of averages. This does not affects the dynamics,
but can lead to a better convergence of the averages.

This subroutine needs to be initialized, as it stores the original forces and force constants to be removed from the dynamics if needed.
"""
mutable struct StochasticSettings{T}
    remove_scha_forces :: Bool
    clean_start_gradient_centroids :: Bool
    clean_start_gradient_fcs :: Bool
    initialized :: Bool
    original_force :: Vector{T}
    original_fc_gradient :: Matrix{T}
end
StochasticSettings(; type=Float64) = StochasticSettings(true, false, false, false, zeros(type, 1), zeros(type, 1, 1))

abstract type GeneralSettings end
get_settings(x :: GeneralSettings) = x.settings

mutable struct ASR{T} <: GeneralSettings
    #evolve_correlators :: Bool
    ignore_small_w :: Bool
    small_w_value :: T
    n_dims :: Int   
    settings :: StochasticSettings
end
ASR(; ignore_small_w=false,
   small_w_value=1e-8, n_dims=3) = ASR(ignore_small_w, small_w_value, n_dims, StochasticSettings())

@doc raw"""
    ASRfixmodes()

This structure is used to fix the acoustic sum rule on modes that are zero.
It will store the eigenvectors of the zeroth modes and constrain the dynamics not to occur along those modes.
"""
mutable struct ASRfixmodes{T} <: GeneralSettings
    small_w_value :: T
    n_dims :: Int
    settings :: StochasticSettings
    eigvect_remove :: Union{Nothing,Matrix{T}}
end
ASRfixmodes(; small_w_value=1e-8, n_dims=3) = ASRfixmodes(small_w_value, n_dims, StochasticSettings(), nothing)

@doc raw"""
    NoASR()

A settings to avoid alltogether the ASR.
"""
struct NoASR <: GeneralSettings
    n_dims :: Int
    settings :: StochasticSettings
end 
NoASR() = NoASR(3, StochasticSettings())


@doc raw"""
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
struct Dynamics{T <: AbstractFloat}
    dt :: T   #In femtosecodns
    total_time :: T # In femtosecodns
    algorithm :: String
    kong_liu_ratio :: T
    verbose :: Bool
    evolve_correlators :: Bool
    seed :: Int64
    N :: Int64
    correlated :: Bool
    
    # Settings
    settings :: GeneralSettings

    # Save the data each
    save_filename :: String 
    save_correlators :: Bool 
    save_each :: Int64
end
Dynamics(dt, total_time, N :: Int; algorithm="generalized-verlet", kong_liu_ratio=1.0, verbose=true, evolve_correlators=true, seed=0, correlated=true, settings=ASR(;n_dims = 3), save_filename="dynamics", save_correlators=false, save_each=100) = Dynamics(dt, total_time, algorithm, kong_liu_ratio, verbose, evolve_correlators, seed, N, correlated, settings, save_filename, save_correlators, save_each)
get_general_settings(x :: Dynamics) = x.settings
get_stochastic_settings(x :: Dynamics) = get_settings(get_general_settings(x))


@doc raw"""
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
    n_atoms :: Int
    n_modes :: Int
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
    
    # settings :: GeneralSettings

    # Eigenvalues and eigenvectors of the current Y matrix 
    λs :: Vector{T}
    λs_vect :: Matrix{T}
    evolve_correlators :: Bool
    cell :: Matrix{T}
    atoms :: Vector{String}
end
function WignerDistribution(n_atoms; type = Float64, n_dims=3, n_modes=n_atoms*n_dims) :: WignerDistribution{type}
    WignerDistribution(R_av = zeros(type, n_atoms*n_dims), 
                       P_av = zeros(type, n_atoms*n_dims), 
                       masses = zeros(type, n_atoms*n_dims), 
                       n_atoms = n_atoms, 
                       n_modes = n_atoms*n_dims, 
                       RR_corr = zeros(type, n_atoms*n_dims, n_atoms*n_dims), 
                       PP_corr = zeros(type, n_atoms*n_dims, n_atoms*n_dims), 
                       RP_corr = zeros(type, n_atoms*n_dims, n_atoms*n_dims), 
                       alpha = zeros(type, n_atoms*n_dims, n_atoms*n_dims), 
                       beta = zeros(type, n_atoms*n_dims, n_atoms*n_dims), 
                       gamma = zeros(type, n_atoms*n_dims, n_atoms*n_dims), 
                       λs = zeros(type, n_modes), 
                       λs_vect = zeros(type, n_atoms*n_dims, n_modes), 
                       evolve_correlators = true, 
                       #settings = ASR(;n_dims = n_dims), 
                       cell = Matrix{type}(I, n_dims, n_dims), 
                       atoms = ["H" for i in 1:n_atoms])
end
get_ndims(rho :: WignerDistribution) = rho.n_modes ÷ rho.n_atoms
get_nmodes(rho :: WignerDistribution) = rho.n_modes
get_natoms(rho :: WignerDistribution) = rho.n_atoms
get_cell(rho :: WignerDistribution) = rho.cell
get_volume(rho :: WignerDistribution) = abs(det(get_cell(rho)))

export get_ndims, get_nmodes, get_natoms



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
    n_configs :: Int
    weights :: Vector{T}
    temperature :: T
    correlated :: Bool
    y0 :: Matrix{T}

    #unit_cell :: Matrix{T}
end 
function Ensemble(wigner_dist :: WignerDistribution{T}, settings :: Dynamics; n_configs :: Int =1, temperature :: Union{Quantity, T} = T(0.0)) :: Ensemble{T} where T
    n_modes = wigner_dist.n_modes
    n_atoms = wigner_dist.n_atoms
    n_dims = n_modes ÷ n_atoms

    # Convert the temperature
    if temperature isa Quantity
        temperature = T(ustrip(uconvert(u"K", temperature)))
    end

    n_modes_y = n_modes
    if !settings.correlated 
        n_modes_y -= get_n_translations(wigner_dist.λs, settings.settings)
    end

    y0 = get_random_y(settings.N, n_modes_y, settings)
    # if !settings.correlated
    #     y0 .*= 0
    # end

    Ensemble(rho0 = wigner_dist,
             positions = zeros(T, n_modes, n_configs),
             energies = zeros(T, n_configs),
             forces = zeros(T, n_modes, n_configs),
             stress = zeros(T, (n_dims * (n_dims+1))÷2, n_configs),
             sscha_energies = zeros(T, n_configs),
             sscha_forces = zeros(T, n_modes, n_configs),
             n_configs = n_configs,
             weights = ones(T, n_configs),
             temperature = temperature,
             correlated = true,
             y0 = y0)
end



"""
Remove acoustic sum rule from eigenvalue and eigenvectors
"""
function remove_translations(vectors, values, thr)
    not_trans_mask =  values .> max(values...) * thr

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

    
    new_values = values[ not_trans_mask ]
    new_vectors = vectors[:, not_trans_mask]


    return new_vectors, new_values
end
function remove_translations(vectors, values; ndims=3)

#    if sum(.!not_trans_mask) != 3
#        println("WARNING")
#        println("the expected number of acustic modes is 3
                #       got $(sum(.!not_trans_mask)) instead")
#        println(values[:3])
#    end
    #@assert sum(.!not_trans_mask) == 3   """
#Error, the expected number of acustic modes is 3
#       got $(sum(.!not_trans_mask)) instead.
#"""

    #new_values = values[ not_trans_mask ]
    #new_vectors = vectors[:, not_trans_mask]
    new_values = values[ndims+1:end]
    new_vectors = vectors[:,ndims+1:end]

    return new_vectors, new_values
end
# == multiple dispatch  == #
function remove_translations(vectors, values, settings :: GeneralSettings) 
	if settings.ignore_small_w
		return remove_translations(vectors, values, settings.small_w_value)
	end
	
	return remove_translations(vectors, values; ndims = settings.n_dims)
end
remove_translations(vect, val, settings :: NoASR) = (vect, val)

function constrain_asr!(matrix_vector, asr :: GeneralSettings) where {T <: AbstractFloat}
end
function constrain_asr!(matrix :: AbstractMatrix{T}, asr :: ASRfixmodes{T}) where {T <: AbstractFloat}
    proj = zeros(T, size(matrix, 1), size(matrix, 2))
    for i in 1:length(asr.eigvect_remove)
        @views proj .+= asr.eigvect_remove[:, i] * asr.eigvect_remove[:, i]'
    end
    matrix .-= proj' * matrix * proj
end
function constrain_asr!(vector :: AbstractVector{T}, asr :: ASRfixmodes{T}) where {T <: AbstractFloat}
    proj = zeros(T, length(vector))
    for i in 1:length(asr.eigvect_remove)
        @views proj .+= asr.eigvect_remove[:, i] * asr.eigvect_remove[:, i]' * vector
    end
    vector .-= proj
end



#function get_n_translations(w_total :: Vector{T<: AbstractFloat}, settings :: GeneralSettings)
function get_n_translations(w_total, settings:: GeneralSettings) :: Int
	if settings.ignore_small_w
		return length(w_total[w_total .< settings.small_w_value])
	end
	return settings.n_dims
end
get_n_translations(w_total, settings :: NoASR) = 0
get_n_translations(settings:: NoASR) = 0



"""
Impose the ASE projecting out the translations
"""
function apply_ASR!(matrix :: Matrix{T}, masses :: Vector{T}) where {T <: AbstractFloat}
    mass_array = zeros(T, size(matrix, 1))
    error("Not implemented")
end

@doc raw"""
    init_from_dyn(dyn :: PyObject, TEMPERATURE :: T, settings :: Dynamics{T}) :: WignerDistribution{T} where {T <: AbstractFloat}
    init_from_dyn(dyn :: PyObject, TEMPERATURE :: Quantity, settings :: Dynamics{T}) :: WignerDistribution{T} where {T <: AbstractFloat}


Initialize a WignerDistribution from a python dynamical matrix.

# Parameters

- `dyn` : PyObject
    The python dynamical matrix
- `TEMPERATURE` : T or Quantity
    The temperature of the system (If float, in kelvin)
- `settings` : Dynamics{T}
    The settings of the dynamics.
"""
function init_from_dyn(dyn :: PyObject, TEMPERATURE :: T, settings :: Dynamics{T}) :: WignerDistribution{T} where {T <: AbstractFloat}

    # Initialize the WignerDistribution structure starting from a dynamical matrix
    
    super_struct = dyn.structure.generate_supercell(dyn.GetSupercell())
    N_modes = Int(super_struct.N_atoms) * 3
    N_atoms = Int(super_struct.N_atoms)

    _w_, pols = dyn.DiagonalizeSupercell() #frequencies are in Ry

    # Convert the frequencies to Ha atomic units
    w = ustrip.(auconvert.(_w_ * u"Ry"))
    
    alpha, beta = get_alphabeta(TEMPERATURE, w, pols, get_general_settings(settings))
    RR_corr, PP_corr = get_correlators(TEMPERATURE, w, pols, get_general_settings(settings))

    gamma = zeros(N_modes, N_modes) #already rescaled (tilde)

    RP_corr = zeros(N_modes, N_modes) #already rescaled (tilde)
    R_av = zeros(T, N_atoms * 3)
    P_av = zeros(T, N_atoms * 3)

    R_av .= reshape(super_struct.coords', :)
    R_av .*= CONV_BOHR # Å to Bohr

    # Get the masses and convert them into atomic units (from Ry)
    masses = super_struct.get_masses_array() 
    masses *= 2

    mass_array = reshape(repeat(masses',3,1), N_modes)

    # Prepare the mass-rescaled quantities
    R_av = R_av.*sqrt.(mass_array)
    P_av = P_av./sqrt.(mass_array)

    # Diagonalize alpha
    if settings.evolve_correlators == false
        lambda_eigen = eigen(alpha)
        λvects, λs = remove_translations(lambda_eigen.vectors, lambda_eigen.values, get_general_settings(settings)) #NO NEEDED WITH ALPHAS
    else
        lambda_eigen = eigen(RR_corr)
        λvects, λs = remove_translations(lambda_eigen.vectors, lambda_eigen.values, get_general_settings(settings)) #NO NEEDED WITH ALPHAS       
    end

    # Cell
    cell = super_struct.unit_cell .*CONV_BOHR
    atoms = super_struct.atoms

    # Initialize the WignerDistribution
    rho = WignerDistribution(N_atoms; n_dims=3, n_modes=length(λs))
    rho.R_av .= R_av
    rho.P_av .= P_av
    rho.RR_corr .= RR_corr
    rho.PP_corr .= PP_corr
    rho.masses .= mass_array

    rho.alpha .= alpha
    rho.beta .= beta
    rho.gamma .= gamma
    rho.evolve_correlators = settings.evolve_correlators
    rho.cell .= cell'
    rho.atoms .= atoms

    #rho = WignerDistribution(R_av  = R_av, P_av = P_av, n_atoms = N_atoms, masses = mass_array, n_modes = N_modes, 
    #                         alpha = alpha, beta = beta, gamma = gamma, RR_corr = RR_corr, PP_corr = PP_corr, RP_corr = RP_corr, 
    #                         λs_vect = λvects, λs = λs, evolve_correlators = settings.evolve_correlators, cell = cell, atoms = atoms)

    update!(rho, settings)
    return rho
end

include("time_evolution.jl")
include("ensemble.jl")
include("phonons.jl")
include("calculator.jl")
include("dynamics.jl")
include("external_f.jl")
include("raman_external_f.jl")

include("UnitfulInterface.jl")

include("symmetry_interface.jl")

export WignerDistribution, get_general_settings,
       NoASR, ASR, integrate!, Dynamics, init_from_dyn,
       get_symmetry_group_from_spglib, get_IR_electric_field,
       Ensemble, single_cycle_pulse, get_IR_electric_field,
       generate_ensemble!, calculate_ensemble!,
       get_volume, get_impulsive_raman_pump,
       get_stochastic_settings, get_settings,
       get_raman_tensor_from_phonons, get_perturbation_direction



end # module QuantumGaussianDynamics
