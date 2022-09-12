

"""
Evolve the Wigner Distribution by a step dt with the Euler algorithm.
You must provide the average derivative of the potential evaluated with the wigner distribution.
"""
function euler_step!(wigner_distribution:: WignerDistribution{T}, dt :: T, dV_dr :: Vector{T}, d2V_dr2 :: Matrix{T}) where {T <: AbstractFloat}
    # Solve the newton equations
    wigner_distribution.P_av .+= -dV_dr .* dt
    wigner_distribution.R_av .+= wigner_distribution.P_av .* dt

    tmp_d2v_mul = similar(d2V_dr2)

    # Evolve the correlators
    wigner_distribution.RP_corr .+= wigner_distribution.PP_corr .* dt

    mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RP_corr)
    tmp_d2v_mul .*= dt

    wigner_distribution.PP_corr .+= tmp_d2v_mul 
    wigner_distribution.PP_corr .+= tmp_d2v_mul'

    mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RR_corr)
    tmp_d2v_mul .*= dt
    wigner_distribution.RP_corr .-=  tmp_d2v_mul

    tmp_d2v_mul .= dt .* wigner_distribution.RP_corr
    wigner_distribution.RR_corr .+= tmp_d2v_mul
    wigner_distribution.RR_corr .+= tmp_d2v_mul'
end 