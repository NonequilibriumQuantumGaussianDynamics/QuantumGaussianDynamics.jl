using Test
using QuantumGaussianDynamics
using Unitful, UnitfulAtomic

const mass = 2.0
const ω = 2.0
const x0 = 0.5

function harmonic_potential!(forces, stress, positions)
    energy = 0 
    stress .= 0.0
    @simd for i in 1:length(positions)
        forces[i] = -mass * ω^2 * (positions[i] - x0)
        energy += 0.5 * mass * ω^2 * (positions[i] - x0)^2
    end
    return energy
end



function test_corr_rr_generate()
    Ncfgs = 100000
    settings = QuantumGaussianDynamics.Dynamics(1.0u"fs", 1.0u"fs", Ncfgs;
                                                settings = NoASR())


    wigner_dist = WignerDistribution(1; n_dims=1)

    # Set to simple
    wigner_dist.RR_corr .= 1.0/(2ω)
    wigner_dist.R_av .= 0.0
    wigner_dist.masses .= mass

    QuantumGaussianDynamics.update!(wigner_dist, settings)
    ensemble = QuantumGaussianDynamics.Ensemble(wigner_dist, settings; n_configs=Ncfgs, temperature=0.0u"K")


    # Generate the ensemble
    QuantumGaussianDynamics.generate_ensemble!(Ncfgs, ensemble, wigner_dist)


    # Compute <RR>
    RR_numerical = 0.0
    for i in 1:Ncfgs
        RR_numerical += ensemble.positions[1, i]^2
    end
    RR_numerical /= Ncfgs

    # println("pos: ", ensemble.positions)
    # println("y: ", ensemble.y0)

    @test RR_numerical ≈ wigner_dist.RR_corr[1,1] atol = 5e-2

    # Compute the forces
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, harmonic_potential!)


    avg_for = zeros(Float64, 1)
    d2v_dr2 = zeros(Float64, 1, 1)
    QuantumGaussianDynamics.get_averages!(avg_for, d2v_dr2, ensemble, wigner_dist)

    # It is mass-rescaled so it does not account for the mass dependency
    @test d2v_dr2[1,1] ≈ ω^2  rtol = 5e-2
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_corr_rr_generate()
end
