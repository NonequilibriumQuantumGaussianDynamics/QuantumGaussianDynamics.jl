using Test
using QuantumGaussianDynamics


@testset "1D ensemble generation" begin 
    include("test_rr_corr.jl")
    test_corr_rr_generate()
end
