
"""
Update the weights of the ensemble according with the new wigner distribution
"""

function update_weights_from_ens!(ensemble:: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T <: AbstractFloat}

    # This routine is used to update the weight after loading the ensemble with load_bin, and the dyn from final_result
    # WARNING: here we use alpha because gamma=0, otherwise we should work with \\alpha-\\gamma^T\\beta^-1\\gamma

    # Prepare the normalization rescale
    lambda_eigen = eigen(wigner_distribution.alpha)
    λvects, λs = remove_translations(lambda_eigen.vectors, lambda_eigen.values) #WARNING, NOT WITH EXTERNAL POTENTIAL
    
    ensemble.weights .= sqrt(λs' * ensemble.λs_inv)
    #wigner_distribution.λs .= λs 
    #wigner_distribution.λs_vect .= λs_vect

    delta_new = zeros(T, wigner_distribution.n_atoms * 3)
    delta_old = zeros(T, wigner_distribution.n_atoms * 3)
    u_new = zeros(T, length(λs))
    u_old = zeros(T, length(λs))


    @views for i = 1:ensemble.n_configs
        delta_new .=  ensemble.positions[:, i] .- wigner_distribution.R_av
        delta_old .=  ensemble.positions[:, i] .- ensemble.original_R_av

        u_new .= lambda_eigen.vectors' * delta_new 
        u_new .*= sqrt(λs)   # using alpha
        u_old .= (ensemble.λs_vect' * delta_old) 
        u_old ./= sqrt(ensemble.λs_inv)

        numerator = - 0.5 * (u_new' * u_new)
        denominator = - 0.5 *  (u_old' * u_old)
        ensemble.weights[i] *= exp( numerator - denominator)
    end
end 


function update_weights!(ensemble:: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T <: AbstractFloat}

    # Check the lenght

    if length(wigner_distribution.λs) != length(ensemble.rho0.λs)
        error("Different length of the eigenvalues vectors")
    end

    # Prepare the normalization rescale
    λs = wigner_distribution.λs
    λs0_inv = ones(T, length(λs)) ./ ensemble.rho0.λs
    
    ensemble.weights .= sqrt(λs' * λs0_inv)
    #wigner_distribution.λs .= λs 
    #wigner_distribution.λs_vect .= λs_vect

    delta_new = zeros(T, wigner_distribution.n_atoms * 3)
    delta_old = zeros(T, wigner_distribution.n_atoms * 3)
    u_new = zeros(T, length(λs))
    u_old = zeros(T, length(λs))


    @views for i = 1:ensemble.n_configs
        delta_new .=  ensemble.positions[:, i] .- wigner_distribution.R_av
        delta_old .=  ensemble.positions[:, i] .- ensemble.rho0.R_av

        u_new .= wigner_distribution.λs_vect' * delta_new 
        u_new .*= sqrt.(λs)   # using alpha
        u_old .= (ensemble.rho0.λs_vect' * delta_old) 
        u_old ./= sqrt.(λs0_inv)

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

    #n_modes = length(wigner_distribution.λs)
    centroid_matrix = repeat(wigner_distribution.R_av,1,ensemble.n_configs)
    d2v_dr2 .= 0

    delta = ensemble.positions .- centroid_matrix
    mul!(d2v_dr2 , delta, ensemble.forces') # sum_{i=1}^{Nconf} delta R_a^i f_c^i

    M = wigner_distribution.alpha - wigner_distribution.gamma' * inv(wigner_distribution.beta) * wigner_distribution.gamma
    d2v_dr2 = -M.*d2v_dr2

    d2v_dr2 .+= d2v_dr2'
    d2v_dr2 ./= 2

    """
    d2v_dr2_tmp = zeros(T, (3*wigner_distribution.n_atoms, 3*wigner_distribution.n_atoms))
    d2v_dr2 .= 0
    @views for i = 1 : ensemble.n_configs
        delta .= ensemble.positions[:, i] - wigner_distribution.R_av
        mul!(d2v_dr2, delta, ensemble.forces', 1.0, 1.0)
    end
    println(d2v_dr2)

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
    """
    # TODO! Impose the acoustic sum rule and the symmetries
end 

"""
Initialize the ensemble
"""
function init_ensemble_from_python(py_ensemble) 

    # Init the ensemble from python

    dyn0 = py_ensemble.current_dyn
    TEMPERATURE = py_ensemble.T0

    rho0 = QuanumGaussianDynamics.init_from_dyn(dyn0, Float64(TEMPERATURE))
    N_atoms = rho0.n_atoms
    N_modes = N_atoms*3

    # Random positions
    ens_positions = reshape(permutedims(py_ensemble.xats,(3,2,1)), (N_modes, py_ensemble.N))
    ens_positions = ens_positions .* CONV_BOHR
    for i in 1:py_ensemble.N
        ens_positions[:,i] = ens_positions[:,i] .* sqrt.(rho0.masses)
    end

    # Random forces
    ens_forces = reshape(permutedims(py_ensemble.forces,(3,2,1)), (N_modes, py_ensemble.N))
    ens_forces = ens_forces .* CONV_RY ./ CONV_BOHR
    ens_forces = ens_forces ./ sqrt.(rho0.masses)

    weights = ones( py_ensemble.N)

    ensemble = QuanumGaussianDynamics.Ensemble(rho0 = rho0, positions = ens_positions, forces = ens_forces, n_configs = Int32(py_ensemble.N),
                                          weights = weights)

    return ensemble
end


function get_λs(RR_corr :: Matrix{T}) where {T <: AbstractFloat}
    eigvals:: T = eigvals(Hermitian(RR_corr))
    return eigvals
end 


# TODO: add a function to load and save the ensemble on disk
function load_ensemble!(ensemble :: Ensemble{T}, path_to_json :: String) where {T <: AbstractFloat}
end
