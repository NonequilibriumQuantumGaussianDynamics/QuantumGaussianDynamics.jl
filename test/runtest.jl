using Test
using QuanumGaussianDynamics

@testset "All Tests" begin
    include("dynamics/main.jl")
    include("zeff/main.jl")
end
