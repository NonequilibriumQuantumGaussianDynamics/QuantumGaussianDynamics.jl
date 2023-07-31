

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

    # Evolve the correlators
    mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RP_corr) # first calculate <d2V><RP>
    tmp_d2v_mul .*= dt

    wigner_distribution.RP_corr .+= wigner_distribution.PP_corr .* dt # update RP

    wigner_distribution.PP_corr .-= Symmetric(tmp_d2v_mul + tmp_d2v_mul')    # update PP
    #wigner_distribution.PP_corr .-= tmp_d2v_mul'      # update PP

    mul!(tmp_d2v_mul, wigner_distribution.RR_corr, d2V_dr2)
    tmp_d2v_mul .*= dt
    wigner_distribution.RP_corr .-=  tmp_d2v_mul      # update RP

    wigner_distribution.RR_corr .+= copy_RP_corr
    wigner_distribution.RR_corr .+= copy_RP_corr'
end 



function euler_step_old!(wigner_distribution:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}) where {T <: AbstractFloat}
    # Solve the newton equations
    wigner_distribution.P_av .+= avg_for .* dt
    wigner_distribution.R_av .+= wigner_distribution.P_av .* dt

    tmp_d2v_mul = similar(d2V_dr2)
    copy_RP_corr = copy(wigner_distribution.RP_corr)
    copy_RP_corr .*= dt

    # Evolve the correlators
    mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RP_corr) # first calculate <d2V><RP>
    tmp_d2v_mul .*= dt

    wigner_distribution.RP_corr .+= wigner_distribution.PP_corr .* dt # update RP

    wigner_distribution.PP_corr .+= tmp_d2v_mul       # update PP
    wigner_distribution.PP_corr .+= tmp_d2v_mul'      # update PP

    mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RR_corr)
    tmp_d2v_mul .*= dt
    wigner_distribution.RP_corr .-=  tmp_d2v_mul      # update RP

    tmp_d2v_mul .= dt .* wigner_distribution.RP_corr
    wigner_distribution.RR_corr .+= copy_RP_corr
    wigner_distribution.RR_corr .+= tmp_d2v_mul'
end 
