push!(LOAD_PATH, "/home/flibbi/programs/sscha/QuantumGaussianDynamics.jl")
println(LOAD_PATH)
import QuanumGaussianDynamics
using QuanumGaussianDynamics.QuanumGaussianDynamics

using PyCall
using LinearAlgebra
#import .QuanumGaussianDynamics

@pyimport cellconstructor.Phonons as PH
@pyimport cellconstructor as CC

#Constants to be moved for later use
#CONV_BOHR = 1.88972598

TEMPERATURE = 0

# Load the dyn corresponding to the equilibrium structure of a SSCHA calculation
sscha_path = "/scratch/flibbi/sscha/tests/usual_folder/"
dyn = PH.Phonons.(sscha_path * "dyn", 10)

super_struct = dyn.structure.generate_supercell(dyn.GetSupercell())
N_modes = Int32(super_struct.N_atoms) * 3
N_atoms = Int32(super_struct.N_atoms)

# Initialization
alpha, beta = dyn.GetAlphaBetaMatrices(float(TEMPERATURE)) #already rescaled (tilde)
gamma = zeros(N_modes, N_modes) #already rescaled (tilde)
R_av = super_struct.coords * CONV_BOHR #units
P_av = zeros(N_atoms, 3)

# Reshape
R_av = reshape(permutedims(R_av), N_modes)
P_av = reshape(permutedims(P_av), N_modes)

# Rescale
masses = super_struct.get_masses_array() # already in Rydberg units
mass_array = vec(repeat(masses,3,1)) # array of 3N masses ordered according to m1 m1 m1 m2 m2 m2 ... 
R_av = R_av.*sqrt.(mass_array)
P_av = P_av./sqrt.(mass_array)

# Initialize
rho = QuanumGaussianDynamics.WignerDistribution(R_av = R_av, P_av = P_av, n_atoms = N_atoms, masses = mass_array, alpha = alpha, beta = beta, gamma = gamma)
println(rho.masses)
