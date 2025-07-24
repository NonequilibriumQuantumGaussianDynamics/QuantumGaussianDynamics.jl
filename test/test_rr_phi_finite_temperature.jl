using TDSCHA
using LinearAlgebra
using Unitful, UnitfulAtomic
using PyCall
using Test

@pyimport cellconstructor
@pyimport cellconstructor.Phonons as PH

"""
This test performs the transformation between <RR> and Φ and checks that the frequencies 
resulting from the transformed Phi matches those of the python cellconstructor
"""
function test_rr_phi()
    fname = joinpath(@__DIR__, "data/dyn_t500_sym_converged")
    dyn_test = PH.Phonons(fname, 1)
    temperature = 1000.0u"K"

    # Get the frequencies from the python object
    ω_ry, pols = dyn_test.DiagonalizeSupercell()

    # Now initialize the wigner
    settings = TDSCHA.Dynamics(0.0, 0.0, 10)
    wigner = TDSCHA.init_from_dyn(dyn_test, temperature, settings)

    Φ = TDSCHA.get_Φ(wigner, ustrip(uconvert(u"K", temperature)))
    ω2_ha = eigen(Φ).values

    println("Frequencies from python: ", ω_ry)
    println("Frequencies from Julia: ", 2 * .√ω2_ha)

    # Loop over the frequencies and check that they match
    for i in 4:length(ω_ry)
        @test ω_ry[i]/2 ≈ √(ω2_ha[i]) atol=1e-8 rtol=1e-5
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_rr_phi()
end
