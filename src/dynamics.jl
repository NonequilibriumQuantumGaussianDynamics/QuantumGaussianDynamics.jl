function integrate!(wigner :: WignerDistribution{T}, ensemble :: Ensemble{T}, settings :: Dynamics{T}) where {T <: AbstractFloat}
    index :: Int32 = 0
    t :: T = 0
    my_dt = settings.dt / CONV_FS # Convert to Rydberg units

    nat3 = wigner.n_atoms* 3

    avg_for = zeros(T, nat3)
    d2v_dr2 = zeros(T, (nat3, nat3))
    
    # Integrate
    while t < settings.total_time
        # Update the ensemble 
        index += 1
        t += settings.dt
        my_dt = settings.dt / CONV_FS

        # Get the average derivatives
        get_averages!(avg_for, d2v_dr2, ensemble, wigner)

        # Integrate
        if "euler" == lowercase(settings.algorithm)
            euler_step!(wigner, my_dt, avg_for, d2v_dr2)
        else
            throw(ArgumentError("""
Error, the selected algorithm $(settings.algorithm)
       is not in the implemented list.
"""))
        end

        # Check if we need to print
        if index % settings%save_each == 0
            if settings.verbose
                println("NEW STEP $index")
                println("==============")
                println()
                println("T = $t fs")
                println()
                println("Average position:")
                println(wigner.R_av)
                println()
            # TODO Save the results on file
            end
        end

        update_weights!(ensemble, wigner)
    end
end 
