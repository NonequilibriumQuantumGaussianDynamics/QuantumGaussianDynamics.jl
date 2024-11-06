using Test
using QuantumGaussianDynamics
using Unitful, UnitfulAtomic


function test_corr_rr_generate()
    Ncfgs = 100000
    settings = QuantumGaussianDynamics.Dynamics(1.0u"fs", 1.0u"fs", Ncfgs;
                                                settings = NoASR())


    wigner_dist = WignerDistribution(1; n_dims=1)

    # Set to simple
    wigner_dist.RR_corr .= 1.0
    wigner_dist.R_av .= 0.0
    wigner_dist.masses .= 1.0

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

    println("RR_numerical = ", RR_numerical)
    @test RR_numerical ≈ wigner_dist.RR_corr[1,1] atol = 5e-2
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_corr_rr_generate()
end
