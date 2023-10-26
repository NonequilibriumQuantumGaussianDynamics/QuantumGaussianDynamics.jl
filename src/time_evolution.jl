

"""
Evolve the Wigner Distribution by a step dt with the Euler algorithm.
You must provide the average derivative of the potential evaluated with the wigner distribution.
"""
function euler_step!(wigner_distribution:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}) where {T <: AbstractFloat}
    # Solve the newton equations
    wigner_distribution.P_av .+= avg_for .* dt
    wigner_distribution.R_av .+= wigner_distribution.P_av .* dt

    tmp_d2v_mul = similar(d2V_dr2)
    copy_RP_corr = copy(wigner_distribution.RP_corr)
    copy_RP_corr .*= dt
    copy_PP_corr = copy(wigner_distribution.PP_corr) # Debug


    # Evolve the correlators
    mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RP_corr) # first calculate <d2V><RP>
    tmp_d2v_mul .*= dt

    wigner_distribution.RP_corr .+= wigner_distribution.PP_corr .* dt # update RP

    wigner_distribution.PP_corr .-= (tmp_d2v_mul .+ tmp_d2v_mul')    # update PP
    #wigner_distribution.PP_corr .-= tmp_d2v_mul'      # update PP

    mul!(tmp_d2v_mul, wigner_distribution.RR_corr, d2V_dr2)
    tmp_d2v_mul .*= dt
    wigner_distribution.RP_corr .-=  tmp_d2v_mul      # update RP
    #println("Check")
    #display(copy_PP_corr.-tmp_d2v_mul )

    wigner_distribution.RR_corr .+= copy_RP_corr
    wigner_distribution.RR_corr .+= copy_RP_corr'
end 


function semi_implicit_euler_step!(wigner_distribution:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}) where {T <: AbstractFloat}
    # Solve the newton equations
    wigner_distribution.P_av .+= avg_for .* dt
    wigner_distribution.R_av .+= wigner_distribution.P_av .* dt

    tmp_d2v_mul = similar(d2V_dr2)
    copy_RP_corr = copy(wigner_distribution.RP_corr)
    copy_RP_corr .*= dt
    copy_PP_corr = copy(wigner_distribution.PP_corr) # Debug


    # Evolve the correlators
    mul!(tmp_d2v_mul, wigner_distribution.RR_corr, d2V_dr2)  # calculate <RR><d2V>
    tmp_d2v_mul .*= dt
    #println("Check")
    #display(wigner_distribution.PP_corr.-tmp_d2v_mul )

    wigner_distribution.RP_corr .+= (wigner_distribution.PP_corr .* dt .- tmp_d2v_mul) # update RP

    mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RP_corr) # calculate <d2V><RP>
    tmp_d2v_mul .*= dt

    wigner_distribution.PP_corr .-= (tmp_d2v_mul .+ tmp_d2v_mul')    # update PP

    wigner_distribution.RR_corr .+= (wigner_distribution.RP_corr .+ wigner_distribution.RP_corr') # update RR
end 


function semi_implicit_verlet_step!(wigner_distribution:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}, part ) where {T <: AbstractFloat}
    # Solve the newton equations
    if part == 1
        wigner_distribution.R_av .+= wigner_distribution.P_av .* dt .+ 1/2.0 * avg_for .* dt^2
        wigner_distribution.P_av .+= 1/2.0 .* avg_for .* dt
    elseif part == 2
        wigner_distribution.P_av .+= 1/2.0 .* avg_for .* dt # repeat with the new force

        tmp_d2v_mul = similar(d2V_dr2)
        copy_RP_corr = copy(wigner_distribution.RP_corr)
        copy_RP_corr .*= dt
        copy_PP_corr = copy(wigner_distribution.PP_corr) # Debug


        # Evolve the correlators
        mul!(tmp_d2v_mul, wigner_distribution.RR_corr, d2V_dr2)  # calculate <RR><d2V>
        tmp_d2v_mul .*= dt
        #println("Check")
        #display(wigner_distribution.PP_corr.-tmp_d2v_mul )

        wigner_distribution.RP_corr .+= (wigner_distribution.PP_corr .* dt .- tmp_d2v_mul) # update RP

        mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RP_corr) # calculate <d2V><RP>
        tmp_d2v_mul .*= dt

        wigner_distribution.PP_corr .-= (tmp_d2v_mul .+ tmp_d2v_mul')    # update PP

        wigner_distribution.RR_corr .+= (wigner_distribution.RP_corr .+ wigner_distribution.RP_corr')*dt # update RR
    end
end 


function generalized_verlet_step!(wigner_distribution:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}, part ) where {T <: AbstractFloat}
    # Solve the newton equations
    if part == 1
        wigner_distribution.R_av .+= wigner_distribution.P_av .* dt .+ 1/2.0 * avg_for .* dt^2
        wigner_distribution.P_av .+= 1/2.0 .* avg_for .* dt

        tmp_d2v_mul = similar(d2V_dr2)
        mul!(tmp_d2v_mul, wigner_distribution.RR_corr, d2V_dr2)  # calculate <RR><d2V>
        wigner_distribution.RR_corr .+= (wigner_distribution.RP_corr .+ wigner_distribution.RP_corr')*dt .- (tmp_d2v_mul .* tmp_d2v_mul').*dt^2/2.0 .+ wigner_distribution.PP_corr .* dt^2

    elseif part == 2
        wigner_distribution.P_av .+= 1/2.0 .* avg_for .* dt # repeat with the new force

        tmp_d2v_mul = similar(d2V_dr2)

        # Evolve the correlators
        mul!(tmp_d2v_mul, wigner_distribution.RR_corr, d2V_dr2)  # calculate <RR><d2V>
        tmp_d2v_mul .*= dt
        #println("Check")
        #display(wigner_distribution.PP_corr.-tmp_d2v_mul )

        wigner_distribution.RP_corr .+= (wigner_distribution.PP_corr .* dt .- tmp_d2v_mul) # update RP

        mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RP_corr) # calculate <d2V><RP>
        tmp_d2v_mul .*= dt

        wigner_distribution.PP_corr .-= (tmp_d2v_mul .+ tmp_d2v_mul')    # update PP

        wigner_distribution.RR_corr .+= (wigner_distribution.RP_corr .+ wigner_distribution.RP_corr') # update RR
    end
end 


function classic_evolution!(Rs::Vector{T}, Ps::Vector{T}, dt::T, cl_for, part) where {T <: AbstractFloat}
        #Ps .+= cl_for .* dt
        #Rs .+= Ps .* dt
        if part == 1
            Rs .+= Ps .* dt + 1/2.0 .* cl_for .* dt^2
            Ps .+= 1/2.0 .* cl_for .* dt
        elseif part ==2
            Ps .+= 1/2.0 .* cl_for .* dt
        end
end

