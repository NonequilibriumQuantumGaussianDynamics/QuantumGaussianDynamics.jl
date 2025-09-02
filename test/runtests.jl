using Test
using QuanumGaussianDynamics

@testset "All Tests" begin
    include("zeff/main.jl")
    include("dynamics/main.jl")
end
