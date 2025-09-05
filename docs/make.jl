using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))   # use local checkout
Pkg.instantiate()

using Documenter, QuantumGaussianDynamics


makedocs(
    sitename = "QuantumGaussianDynamics.jl",
    format   = Documenter.HTML(prettyurls = false),
    modules  = [QuantumGaussianDynamics],
    clean    = true,
    pages    = [
        "Home" => "index.md",
	"API"  => "api.md",
    ],
    build    = joinpath(@__DIR__, "build"),
)

deploydocs(
    repo = "github.com/yourusername/QuantumGaussianDynamics.jl.git",
    devbranch = "main",   # or "master" if that's your default branch
)

