using QuantumGaussianDynamics
using Test

using Unitful, UnitfulAtomic

const x0 = 0.1
const ω = 1.0
const mass = 1.0

function harmonic_potential!(forces, stress, positions)
    energy = 0 
    stress .= 0.0
    @simd for i in 1:length(positions)
        forces[i] = -mass * ω^2 * (positions[i] - x0)
        energy += 0.5 * mass * ω^2 * (positions[i] - x0)^2
    end
    return energy
end


function test_harmonic()

    algorithm = "generalized-verlet"
    dt = 0.01u"fs"
    total_time = 10.0u"fs"
    N_configs = 10000

    # We do not want ASR to be imposed
    settings = QuantumGaussianDynamics.Dynamics(dt, total_time, N_configs;
                                                algorithm = algorithm,
                                                settings = NoASR(),
                                                save_each=1)

    wigner_dist = WignerDistribution(1; n_dims=1)
    
    # Generate a good initial distribution
    wigner_dist.RR_corr .= 1/(2ω)
    wigner_dist.PP_corr .= ω^2 * wigner_dist.RR_corr
    wigner_dist.RP_corr .= 0.0
    wigner_dist.R_av .= 0.0
    wigner_dist.P_av .= 0.0
    wigner_dist.masses .= mass
    QuantumGaussianDynamics.update!(wigner_dist, settings)

    ensemble = QuantumGaussianDynamics.Ensemble(wigner_dist, settings; n_configs=100, temperature=0.0u"K")
    efield = QuantumGaussianDynamics.fake_field(1; ndims=1)

    QuantumGaussianDynamics.generate_ensemble!(N_configs, ensemble, wigner_dist)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, harmonic_potential!)

    QuantumGaussianDynamics.integrate!(wigner_dist, ensemble, settings, harmonic_potential!, efield)

    @test wigner_dist.RR_corr[1,1] ≈ 1/(2ω) rtol = 1e-2
end


if abspath(PROGRAM_FILE) == @__FILE__
    test_harmonic()
end
