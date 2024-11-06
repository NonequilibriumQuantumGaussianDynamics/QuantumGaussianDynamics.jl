function integrate!(wigner :: WignerDistribution{T}, ensemble :: Ensemble{T}, settings :: Dynamics{T}, crystal, efield :: ElectricField{T}; symmetry_group :: Symmetries{T} = get_empty_symmetry_group(T)) where T
    
    index :: Int32 = 0
    t :: T = 0
    my_dt = settings.dt / CONV_FS # Convert to Rydberg units

    nat3 = wigner.n_atoms * get_ndims(wigner)

    avg_for = zeros(T, nat3)
    d2v_dr2 = zeros(T, (nat3, nat3))

    Rs = deepcopy(wigner.R_av)
    Ps = deepcopy(wigner.P_av)

    name = settings.save_filename*"$(settings.dt)-$(settings.total_time)-$(settings.N)"

    rank = 0
    size = 1
    if MPI.Initialized()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        size = MPI.Comm_size(MPI.COMM_WORLD)
    end

    file0 = init_file(name*".pos")
    file1 = init_file(name*".rho")
    file2 = init_file(name*".for")
    file3 = init_file(name*".cl")
    file4 = init_file(name*".ext")
    file5 = init_file(name*".str")
    file6 = init_file(name*".bet")
    file7 = init_file(name*".gam")
    
    # Get the average derivatives
    get_averages!(avg_for, d2v_dr2, ensemble, wigner)
    total_energy = get_total_energy(ensemble, wigner)
    classic_for = get_classic_forces(wigner, crystal)
    cl_energy, cl_for = get_classic_ef(Rs, wigner, crystal)
    ext_for = get_external_forces(t/CONV_FS, efield, wigner)

    tot_for = avg_for .+ ext_for
    tot_cl_for = cl_for .+ ext_for

    println("Forces ", tot_for)
    println("Classic forces ", tot_cl_for)
    println("External forces ", ext_for)
    println("D2V_DR2 ", d2v_dr2)

    
    # Impose symmetries
    if !isempty(symmetry_group)
        symmetry_group.symmetrize_vector!(tot_for)
        symmetry_group.symmetrize_vector!(tot_cl_for)
        symmetry_group.symmetrize_fc!(d2v_dr2)
    end

    println("After symmetries")
    println("Forces ", tot_for)
    println("Classic forces ", tot_cl_for)
    println("External forces ", ext_for)
    println("D2V_DR2 ", d2v_dr2)




    # Integrate
    while t < settings.total_time
        # Update the ensemble 
        index += 1
        t += settings.dt
        my_dt = settings.dt / CONV_FS

        # Integrate
        if "euler" == lowercase(settings.algorithm)
            euler_step!(wigner, my_dt, tot_for, d2v_dr2)
        elseif "semi-implicit-euler" == lowercase(settings.algorithm)
            semi_implicit_euler_step!(wigner, my_dt, tot_for, d2v_dr2)
        elseif "semi-implicit-verlet" == lowercase(settings.algorithm)
            semi_implicit_verlet_step!(wigner, my_dt, tot_for, d2v_dr2, 1)
        elseif "fixed" == lowercase(settings.algorithm)
            fixed_step!(wigner, my_dt, tot_for, d2v_dr2, 1)
        elseif "generalized-verlet" == lowercase(settings.algorithm) 
            bc0 = get_merged_vector(wigner.PP_corr, wigner.RP_corr)
            generalized_verlet_step!(wigner, my_dt, tot_for, d2v_dr2, bc0, 1) 
        elseif "none" == lowercase(settings.algorithm)
            nothing

        else
            throw(ArgumentError("""
Error, the selected algorithm $(settings.algorithm)
       is not in the implemented list.
"""))
        end

        # Classic integration (part 1)
        classic_evolution!(Rs, Ps, my_dt, tot_cl_for, 1)

        println("R_av ", wigner.R_av)
        println("P_av ", wigner.P_av)
        println("dt = $my_dt - dt^2 = $(my_dt^2)")

        # Update the eigenvalues of the Wigner matrix
        update!(wigner, get_general_settings(settings))

        # Update the stochastic weights
        update_weights!(ensemble, wigner)
        kl = get_kong_liu(ensemble)
        if rank == 0
            println("KL ratio ", kl/T(ensemble.n_configs))
        end
        if kl < settings.kong_liu_ratio*ensemble.n_configs
            generate_ensemble!(settings.N,ensemble, wigner)
            calculate_ensemble!(ensemble, crystal)
        end

        # Get the average derivatives
        # TODO: Too many allocating functions
        get_averages!(avg_for, d2v_dr2, ensemble, wigner)
        total_energy = get_total_energy(ensemble, wigner)
        avg_stress = get_average_stress(ensemble, wigner)
        classic_for = get_classic_forces(wigner, crystal)
        cl_energy, cl_for = get_classic_ef(Rs, wigner, crystal)
        ext_for = get_external_forces(t/CONV_FS, efield, wigner)

        tot_for = avg_for .+ ext_for
        tot_cl_for = cl_for .+ ext_for

        
        # Impose symmetries
        if !isempty(symmetry_group)
            symmetry_group.symmetrize_vector!(tot_for)
            symmetry_group.symmetrize_vector!(tot_cl_for)
            symmetry_group.symmetrize_fc!(d2v_dr2)
        end


        if "semi-implicit-verlet" == lowercase(settings.algorithm)
            semi_implicit_verlet_step!(wigner, my_dt, tot_for, d2v_dr2, 2)
        elseif "fixed" == lowercase(settings.algorithm)
            fixed_step!(wigner, my_dt, tot_for, d2v_dr2, 2)
        elseif "generalized-verlet" == lowercase(settings.algorithm) 
            generalized_verlet_step!(wigner, my_dt, tot_for, d2v_dr2, bc0, 2)
        end
        # Classic integration (part 2)
        classic_evolution!(Rs, Ps, my_dt, tot_cl_for, 2)

        # Check if we need to print
        if index % settings.save_each == 0
            if settings.verbose
                if rank == 0
                    println("NEW STEP $index")
                    println("==============")
                    println()
                    println("T = $t fs")
                    println()
                end
            #println("Average position:")
                #println(wigner.R_av)
                #println()
                #println("Average momenta:")
                #println(wigner.P_av)
                #println()
                ## Positions are exported in a.u. (Bohr)
                line = "$t "                            
                for i in 1:nat3
                    line *= "  $(wigner.R_av[i]/sqrt(wigner.masses[i])) "
                end
                for i in 1:nat3
                    line *= "  $(wigner.P_av[i]*sqrt(wigner.masses[i])) "
                end
                if rank==0
                    println("! $t $(wigner.R_av) $(wigner.P_av) $(wigner.masses) $total_energy")
                end
                line *= " $total_energy"
                line *= "\n"
                write_file(file0,line)

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
                write_file(file2, line)                   


                line = ""
                for i in 1:nat3 , j in 1:nat3
                    line *= "  $(wigner.RR_corr[i,j]/sqrt(wigner.masses[i])/sqrt(wigner.masses[j])) "
                end
                line *= "\n"
                write_file(file1,line)

                line = ""
                for i in 1:nat3 , j in 1:nat3
                    line *= "  $(wigner.PP_corr[i,j]*sqrt(wigner.masses[i])*sqrt(wigner.masses[j])) "
                end
                line *= "\n"
                write_file(file6,line)

                line = ""
                for i in 1:nat3 , j in 1:nat3
                    line *= "  $(wigner.RP_corr[i,j]/sqrt(wigner.masses[i])*sqrt(wigner.masses[j])) "
                end
                line *= "\n"
                write_file(file7,line)


                line = ""
                for i in 1:nat3
                    line *= "  $(Rs[i]/sqrt(wigner.masses[i])) "
                end
                for i in 1:nat3
                    line *= "  $(Ps[i]*sqrt(wigner.masses[i])) "
                end
                line *= " $cl_energy"
                line *= "\n"
                write_file(file3,line)

                line = ""
                for i in 1:nat3
                    line *= "  $(ext_for[i]*sqrt(wigner.masses[i])) "
                end
                line *= "\n"
                write_file(file4,line)


                line = ""
                for i in 1:nat3 * (nat3+1) ÷ 2
                    line *= " $(avg_stress[i]) "
                end
                line *= "\n"
                write_file(file5,line)

            # TODO Save the results on file
            end
        end

        #if index%500 == 0
            #println("Garbage")
            #GC.gc()
        #end
    end
    if rank==0
        close(file0)
        close(file1)
        close(file2)
        close(file3)
        close(file4)
        close(file5)
        close(file6)
        close(file7)
    end
end 
