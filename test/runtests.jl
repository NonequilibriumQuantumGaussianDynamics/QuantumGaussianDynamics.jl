using Test
using QuantumGaussianDynamics


@testset "1D ensemble generation" begin 
    include("test_rr_corr.jl")
    test_corr_rr_generate()
end

@testset "1D harmonic oscillator" begin 
    include("test_harmonic.jl")
    test_harmonic()
end

@testset "2D harmonic oscillator" begin 
    include("test_harmonic_2d.jl")
    test_harmonic_1_particle_2d()
end

@testset "ASE calculator" begin 
    include("test_ase_calculator.jl")
    test_compute_force_ase()

    wj = test_julia_harmonic3d()
    wp = test_ase_calculator_harmonic()

    for i in 1:3
        @test wj.R_av[i] ≈ wp.R_av[i] rtol = 5e-2
        @test wj.P_av[i] ≈ wp.P_av[i] rtol = 5e-2

        @test wj.RR_corr[i,i] ≈ wp.RR_corr[i,i] rtol = 5e-2
        @test wj.RP_corr[i,i] ≈ wp.RP_corr[i,i] rtol = 5e-2
        @test wj.PP_corr[i,i] ≈ wp.PP_corr[i,i] rtol = 5e-2
    end
end
