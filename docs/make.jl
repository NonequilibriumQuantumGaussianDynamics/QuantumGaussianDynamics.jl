using Documenter
using QuanumGaussianDynamics
using Unitful

makedocs(
    sitename = "QuantumGaussianDynamics.jl",
    modules = [QuanumGaussianDynamics],
    pages = [
        "Home" => "index.md",
    ],
)
