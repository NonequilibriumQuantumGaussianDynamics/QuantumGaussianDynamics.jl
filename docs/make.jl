using Documenter, QuantumGaussianDynamics

makedocs(
    sitename = "QuantumGaussianDynamics.jl",
    format   = Documenter.HTML(),
    modules  = [QuantumGaussianDynamics],
    clean    = true,
    strict   = true,          # fail the build on doc warnings (nice for CI)
    pages    = [
        "Home" => "index.md",
        "API"  => "api.md",
    ],
)

# For GitHub Pages deployment
deploydocs(
    repo = "https://github.com/NonequilibriumQuantumGaussianDynamics/QuantumGaussianDynamics.jl",
    devbranch = "main",       # or "master"
