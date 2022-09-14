
"""
Update the weights of the ensemble according with the new wigner distribution
"""
function update_weights!(ensemble:: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T :< AbstractFloat}
    # Prepare the normalization rescale
    lambda_eigen = eigen(wigner_distribution.RR_corr)

    # Discard the sum rule
    λvects, λs = remove_translations(lambda_eigen.vectors, lambda_eigen.values)
    ensemble.weights .= sqrt(λs' * ensemble.λs_inv)
    wigner_distribution.λs .= λs 
    wigner_distribution.λs_vect .= λs_vect


    delta_new = zeros(T, wigner_distribution.n_atoms * 3)
    delta_old = zeros(T, wigner_distribution.n_atoms * 3)
    u_new = zeros(T, length(λs))
    u_old = zeros(T, length(λs))


    @views for i = 1:ensemble.n_configs
        delta_new .=  ensemble.positions[:, i] .- wigner_distribution.R_av
        delta_old .=  ensemble.positions[:, i] .- ensemble.original_R_av

        u_new .= lambda_eigen.vectors' * delta_new 
        u_new ./= sqrt(λs)
        u_old .= (ensemble.λs_vect' * delta_old) 
        u_old .*= sqrt(ensemble.λs_inv)

        numerator = - 0.5 * (u_new' * u_new)
        denominator = - 0.5 *  (u_old' * u_old)
        ensemble.weights[i] *= exp( numerator - denominator)
    end
end 


"""
Evaluate the average dv_dr and d2v_dr2 from the ensemble and the wigner distribution.

The weights on the ensemble should have been already updated.
"""
function get_averages!(dv_dr :: Vector{T}, d2v_dr2 :: Matrix{T}, ensemble :: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T <: AbstractFloat}
    dv_dr .= sum(ensemble.forces, dims = 2)

    n_modes = length(wigner_distribution.λs)
    delta = zeros(T, wigner_distribution.n_atoms * 3)
    d2v_dr2_tmp = zeros(T, (3*wigner_distribution.n_atoms, 3*wigner_distribution.n_atoms))
    d2v_dr2 .= 0
    @views for i = 1 : ensemble.n_configs
        delta .= ensemble.positions[:, i] - wigner_distribution.R_av
        mul!(d2v_dr2, delta, ensemble.forces', 1.0, 1.0)
    end

    e_tilde = copy(wigner_distribution.λs_vect)
    for i = 1 : n_modes
        e_tilde[:, i] .*= wigner_distribution.λs
    end 

    # Now multiply by upsilon
    mul!(d2v_dr2_tmp, e_tilde', d2v_dr2)
    mul!(d2v_dr2, wigner_distribution.λs_vect, d2v_dr2)

    # Impose permutation symmetry
    d2v_dr2 .+= d2v_dr2'
    d2v_dr2 ./= 2

    # TODO! Impose the acoustic sum rule and the symmetries
end 

"""
Initialize the ensemble
"""
function init_ensemble!(ensemble :: Ensemble{T}) where {T <: AbstractFloat}
    # Discard the sum rule (TODO, improve the sum rule on the eigenvectors)
    ensemble.λs_vect, λs = remove_translations(lambda_eigen.vectors, lambda_eigen.values)
    ensemble.λs_inv = 1 ./ λs
end


function get_λs(RR_corr :: Matrix{T}) where {T <: AbstractFloat}
    eigvals:: T = eigvals(Hermitian(RR_corr))
    return eigvals
end 


# TODO: add a function to load and save the ensemble on disk
function load_ensemble!(ensemble :: Ensemble{T}, path_to_json :: String) where {T <: AbstractFloat}
end
