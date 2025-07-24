using MPI
using PyCall
using Unitful, UnitfulAtomic
using Random
using TDSCHA

@pyimport ase

const ω1 = 500.0u"c/cm" * 2π
const ω2 = 700.0u"c/cm" * 2π
const ω3 = 900.0u"c/cm" * 2π
const halfmass = 918.0u"me" 

# function harmonic_calculator!(forces, stress, positions)
#     energy = 0 
#     stress .= 0.0
#     ω = [ω1, ω2, ω3]
#     k = ustrip.(auconvert.(halfmass * ω.^2))
#     @simd for i in 1:length(positions)
#         forces[i] = -k[i] * positions[i]
#         energy += 0.5 * k[i] * positions[i]^2
#     end
#     return energy
# end
# function harmonic_calculator_asr!(forces, stress, positions)
#     energy = 0 
#     stress .= 0.0
#     ω = [ω1, ω2, ω3]
#     k = ustrip.(auconvert.(halfmass * ω.^2))
# 
#     δcoords = positions[1:3] .- positions[4:end]
# 
#     @simd for i in 1:3
#         δ = positions[i] - positions[i+3]
#         forces[i] = -k[i] * δ
#         forces[i+3] = -forces[i]
#         energy += 0.5 * k[i] * δ^2
#     end
#     return energy
# end

function test_compute_force_MPI()
    # Initialize MPI
    MPI.Init()

    # Apply the seed to allow for reproducibility
    Random.seed!(1234)

    N_configs = 8
    dt = 0.001u"ps"
    total_time = 0.1u"ps"

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    size = MPI.Comm_size(MPI.COMM_WORLD)

    rank != 0 ? redirect_stdout(open("/dev/null", "w")) : redirect_stdout(stdout)

    if rank == 0
        println("Running test_compute_force_MPI with ", size, " processes")
    end

    # Import a local python module
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    ase_calc_module = pyimport("test_ase_calculator")

    # Convert the force constant in ASE units
    k1 = uconvert(u"eV/Å^2", ω1^2 * halfmass)
    k2 = uconvert(u"eV/Å^2", ω2^2 * halfmass)
    k3 = uconvert(u"eV/Å^2", ω3^2 * halfmass)


    py_calc = ase_calc_module.Harmonic3D(ustrip(k1), 
                                         ustrip(k2), 
                                         ustrip(k3))


    settings = TDSCHA.Dynamics(dt, total_time, N_configs;
                               settings = NoASR())
                                               


    wigner = WignerDistribution(1; n_dims=3)
    calculator = TDSCHA.init_calculator(py_calc, wigner, ase.Atoms)

    wigner.RR_corr[1,1] = ustrip(auconvert(1/(2ω1)))
    wigner.RR_corr[2,2] = ustrip(auconvert(1/(2ω2)))
    wigner.RR_corr[3,3] = ustrip(auconvert(1/(2ω3)))
    wigner.PP_corr[1,1] = ustrip(auconvert(ω1/2))
    wigner.PP_corr[2,2] = ustrip(auconvert(ω2/2))
    wigner.PP_corr[3,3] = ustrip(auconvert(ω3/2))

    wigner.R_av .= ustrip(auconvert(0.1u"Å"))

    wigner.masses .= ustrip(auconvert(halfmass))
    wigner.atoms .= "H"

    TDSCHA.update!(wigner, settings)

    ensemble = TDSCHA.Ensemble(wigner, settings; n_configs=N_configs, temperature=0.0u"K")

    TDSCHA.generate_ensemble!(ensemble, wigner)
    TDSCHA.calculate_ensemble!(ensemble, calculator)

    # Print the forces on the configurations
    if rank == 0
        for i in 1:ensemble.n_configs
            println("Configuration $i: r⃗ = $(ensemble.positions[:, i])  f⃗ = $(ensemble.forces[:, i])")
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    test_compute_force_MPI()
end

