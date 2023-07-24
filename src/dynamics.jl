function integrate!(wigner :: WignerDistribution{T}, ensemble :: Ensemble{T}, settings :: Dynamics{T}, time :: T) where {T <: AbstractFloat}
    index :: Int32 = 0
    t :: T = 0
    my_dt = settings.dt * CONV_FS # Convert to inner units

    nat3 = wigner.n_atoms* 3

    dv_dr = zeros(T, nat3)
    d2v_dr2 = zeros(T, (nat3, nat3))
    
    # Integrate
    while t < time
        # Update the ensemble 
        index += 1
        t += dt

        update_weights!(ensemble, wigner)

        # Get the average derivatives
        get_averages!(dv_dr, d2v_dr2, ensemble, wigner)

        # Integrate
        if "euler" == lowercase(settings.algorithm)
            euler_step!(wigner, my_dt, dv_dr, d2v_dr2)
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
end 
