using Test
using QuantumGaussianDynamics

@testset "Zeff" begin
    include("zeff/main.jl")
end

@testset "Dynamics" begin
    include("dynamics/SIE.jl")
    include("dynamics/GV.jl")
end

@testset "Stress" begin
    include("stress/main.jl")
end
