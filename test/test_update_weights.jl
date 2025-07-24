using TDSCHA
using Test

using Unitful, UnitfulAtomic
using PyCall


function test_update_weights(;verbose=false)
    # Import the cellconstructor
    pyimport("cellconstructor")
    pyimport("sscha")
    pyphonons = pyimport("cellconstructor.Phonons")
    pyensemble = pyimport("sscha.Ensemble")

    temperature = 0.0u"K"

    dyn = pyphonons.Phonons(joinpath(@__DIR__, "../examples/H2_EMTpot/sscha_ensemble/dyn_gen_pop1_"))
    novel_dyn = pyphonons.Phonons(joinpath(@__DIR__, "../examples/H2_EMTpot/sscha_ensemble/dyn_gen_pop1_"))

    ens_py = pyensemble.Ensemble(dyn, ustrip(temperature))
    ens_py.load_bin(joinpath(@__DIR__, 
                             "../examples/H2_EMTpot/sscha_ensemble"), 1)
    mask = [i < 5 for i in 1:ens_py.N]
    ens_py = ens_py.split(mask)
    # for i in 1:ens_py.N
    #     for j in 1:3
    #         ens_py."sscha_forces"[i-1, j-1] = 0
    #     end
    # end

    # Get the QGD process
    settings = Dynamics(0.01u"fs", 1.0u"fs", ens_py.N)
    set_remove_scha_forces!(settings, true)

    rho = TDSCHA.init_from_dyn(dyn, temperature, settings)
    ensemble = TDSCHA.init_ensemble_from_python(ens_py, settings)

    # Lets compute the force without reweighting
    avg_for_py = reshape(permutedims(ens_py.get_average_forces(false), (2,1)), :)
    avg_for_py = ustrip.(auconvert.(avg_for_py .* u"Ry/Å"))
    avg_for_py ./= sqrt.(rho.masses)

 
    avg_for_qgd = zeros(Float64, get_natoms(rho) * 3)
    d2dumb = zeros(Float64, get_natoms(rho)*3, get_natoms(rho)*3)

    TDSCHA.get_averages!(avg_for_qgd, d2dumb, ensemble,
                                          rho, get_stochastic_settings(settings))

    if verbose
        println("Test average force before reweighting")
        println("PY", avg_for_py)
        println("JL", avg_for_qgd)
    end
    for i in 1:3get_natoms(rho)
        @test avg_for_py[i] ≈ avg_for_qgd[i]
    end


    # Let us change the dynamical matrix and see if it works
    @show novel_dyn.structure.coords
    novel_dyn.structure."coords"[1,1] = novel_dyn.structure.coords[1,1]*1.01
    novel_dyn.structure."coords"[2,1] = novel_dyn.structure.coords[2,1]*1.01

    @show novel_dyn.structure.coords

    ens_py.update_weights(novel_dyn, ustrip(temperature))
    
    # Get the average force
    avg_for_py = ens_py.get_average_forces(false)
    #
    # Check the average of the forces
    avg_for_py = reshape(permutedims(ens_py.get_average_forces(false), (2,1)), :)
    avg_for_py = ustrip.(auconvert.(avg_for_py .* u"Ry/Å"))
    avg_for_py ./= sqrt.(rho.masses)

    # Get the novel density
    rho_new = TDSCHA.init_from_dyn(novel_dyn, temperature, settings)

    # Update the weights in julia
    TDSCHA.update_weights!(ensemble, rho_new)
    TDSCHA.get_averages!(avg_for_qgd, d2dumb, ensemble,
                                          rho_new, get_stochastic_settings(settings))

    # Check the consistency of the sscha forces
    for k in 1:ens_py.N
        for i in 1:get_natoms(rho)
            for j in 1:3
                test_frc = ens_py.sscha_forces[k, i, j]

                # Convert in Ha/Bohr
                test_frc = ustrip(auconvert(test_frc * u"Ry/Å"))

                # Divide by the sqrt of mass
                test_frc /= √rho.masses[3i]

                @test test_frc ≈ ensemble.sscha_forces[3(i-1)+j, k] rtol = 1e-4
                println("TEST SCHA F $i $j > ", test_frc, ensemble.sscha_forces[3(i-1)+j, k])

            end
        end
    end


    # Check the update
    for i in 1:get_nconfigs(ensemble)
        @test ensemble.weights[i] ≈ ens_py.rho[i] rtol=1e-4
        if verbose
            println("Test weights: $(ensemble.weights[i]), $(ens_py.rho[i])")
        end
    end

    # Check the average of the forces
    if verbose
        println("Test average force after reweighting")
        println("PY", avg_for_py)
        println("JL", avg_for_qgd)
    end
    for i in 1:3get_natoms(rho)
        @test avg_for_py[i] ≈ avg_for_qgd[i] rtol = 1e-4
    end



    # Perform an update also on the dynamical matrix
    for i in 1:6
        for j in 1:6
            novel_dyn."dynmats"[1][i, j] = novel_dyn."dynmats"[1][i, j] * 1.005
        end
    end

    ens_py.update_weights(novel_dyn, ustrip(temperature))
    
    # Get the novel density
    rho_new = TDSCHA.init_from_dyn(novel_dyn, temperature, settings)

    # Update the weights in julia
    TDSCHA.update_weights!(ensemble, rho_new)

    # Check the update
    for i in 1:get_nconfigs(ensemble)
        @test ensemble.weights[i] ≈ ens_py.rho[i] rtol=1e-4
        if verbose
            println("Test weights 2: $(ensemble.weights[i]), $(ens_py.rho[i])")
        end
    end

    
end


if abspath(PROGRAM_FILE) == @__FILE__
    test_update_weights(;verbose=true)
end
