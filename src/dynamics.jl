function integrate!(wigner :: WignerDistribution{T}, ensemble :: Ensemble{T}, settings :: Dynamics{T}, crystal) where {T <: AbstractFloat}
    index :: Int32 = 0
    t :: T = 0
    my_dt = settings.dt / CONV_FS # Convert to Rydberg units

    nat3 = wigner.n_atoms* 3

    avg_for = zeros(T, nat3)
    d2v_dr2 = zeros(T, (nat3, nat3))

    name = settings.save_filename*"$(settings.dt)-$(settings.total_time)-$(settings.N)"
    file = open(name*".pos", "w")
    close(file)
    file = open(name*".pos", "a")
    file1 = open(name*".rho", "w")
    close(file1)
    file1 = open(name*".rho", "a")
    file2 = open(name*".for", "w")
    close(file2)
    file2 = open(name*".for", "a")
    
    # Integrate
    while t < settings.total_time
        # Update the ensemble 
        index += 1
        t += settings.dt
        my_dt = settings.dt / CONV_FS

        # Get the average derivatives
        get_averages!(avg_for, d2v_dr2, ensemble, wigner)
        println("Average forces:")
        println(avg_for)
        println("d2Vdr")
        display(d2v_dr2./CONV_RY.*CONV_BOHR^2.0*wigner.masses[1])
        classic_for = get_classic_forces(wigner, crystal)


        # Integrate
        if "euler" == lowercase(settings.algorithm)
            """
            println("before RR")
            println(wigner.RR_corr)
            println("before PP")
            println(wigner.PP_corr)
            println("before RP")
            println(wigner.PP_corr)
            println("isimmutable ",isimmutable(wigner.RR_corr) )
            """
            euler_step!(wigner, my_dt, avg_for, d2v_dr2)
            """
            println("after RR")
            println(wigner.RR_corr)
            println("after PP")
            println(wigner.PP_corr)
            println("after RP")
            println(wigner.RP_corr)
            """
        elseif "semi-implicit-euler" == lowercase(settings.algorithm)
            semi_implicit_euler_step!(wigner, my_dt, avg_for, d2v_dr2)

        else
            throw(ArgumentError("""
Error, the selected algorithm $(settings.algorithm)
       is not in the implemented list.
"""))
        end

        # Check if we need to print
        if index % settings.save_each == 0
            if settings.verbose
                println("NEW STEP $index")
                println("==============")
                println()
                println("T = $t fs")
                println()
                println("Average position:")
                println(wigner.R_av)
                println()
                println("Average momenta:")
                println(wigner.P_av)
                println()
                line = "$t "                            
                for i in 1:nat3
                    line *= "  $(wigner.R_av[i]/sqrt(wigner.masses[i])) "
                end
                for i in 1:nat3
                    line *= "  $(wigner.P_av[i]*sqrt(wigner.masses[i])) "
                end
                line *= "\n"
                if rank==root
                    write(file, line)                   
                end

                line = ""
                for i in 1:nat3
                    line *= "  $(avg_for[i]*sqrt(wigner.masses[i])) "
                end
                for i in 1:nat3
                    line *= "  $(classic_for[i]*sqrt(wigner.masses[i])) "
                end
                for i in 1:nat3 , j in 1:nat3
                    line *= "  $(d2v_dr2[i,j]*sqrt(wigner.masses[i])*sqrt(wigner.masses[j]))"
                end
                line *= "\n"
                if rank==root
                    write(file2, line)                   
                end

                line = ""
                for i in 1:nat3 , j in 1:nat3
                    println("line $i $j")
                    line *= "  $(wigner.RR_corr[i,j]/sqrt(wigner.masses[i])/sqrt(wigner.masses[j])) "
                end
                println(line)
                #for i in 1:nat3 , j in 1:nat3
                #    line *= "  $(wigner.PP_corr[i,j]) "
                #end
                #for i in 1:nat3 , j in 1:nat3
                #    line *= "  $(wigner.RP_corr[i,j]) "
                #end
                line *= "\n"
                if rank == root
                    write(file1,line)
                end

            # TODO Save the results on file
            end
        end
        println("RR")
        println(wigner.RR_corr)
        lambda_eigen = eigen(Symmetric(wigner.RR_corr))
        λvects, λs = QuanumGaussianDynamics.remove_translations(lambda_eigen.vectors, lambda_eigen.values)
        println("before")
        println("w ", wigner.λs)
        println("e ", ensemble.rho0.λs)
        wigner.λs_vect = λvects
        wigner.λs = λs
        println("after")
        println("w ", wigner.λs)
        println("e ", ensemble.rho0.λs)


        update_weights!(ensemble, wigner)
        kl = get_kong_liu(ensemble)
        println("KL ratio ", kl/T(ensemble.n_configs))
        if kl < settings.kong_liu_ratio*ensemble.n_configs
            generate_ensemble!(settings.N,ensemble, wigner)
            calculate_ensemble!(ensemble, crystal)
        end
        println("weights")
        #println(ensemble.weights)
        #println("len, ", length(ensemble.weights))
    end
    close(file)
    close(file1)
    close(file2)
end 
