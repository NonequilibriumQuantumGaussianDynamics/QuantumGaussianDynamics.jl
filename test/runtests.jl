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
