
"""
Update the weights of the ensemble according with the new wigner distribution
"""
function update_weights!(ensemble:: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T :< AbstractFloat}
    # Prepare the normalization rescale
    λs = get_λs(wigner_distribution.RR_corr)
    ensemble.weights .= sqrt(λs' * ensemble.λs_inv)

    # TODO: It is possible to get everything more efficient and avoid these two inversions
    #       by simply storing also the eigenvectors and projecting delta into the space on which Upsilon is diagonal
    upsilon_new = inv(wigner_distribution.RR_corr)
    upsilon_old = inv(wigner_distribution)
    delta_new = ensemble.positions - wigner_distribution.R_av
    delta_old = ensemble.positions - ensemble.original_R_av
end 


"""
Initialize the ensemble
"""
function init_ensemble!(ensemble :: Ensemble)
    ensemble.λs_inv = 1 ./ get_λs(ensemble.original_RR_corr)
end


function get_λs(RR_corr :: Matrix{T}) where {T <: AbstractFloat}
    eigvals:: T = eigvals(Hermitian(RR_corr))
    return eigvals
end 