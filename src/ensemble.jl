
"""
Update the weights of the ensemble according with the new wigner distribution
"""


function update_weights!(ensemble:: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T <: AbstractFloat}

    # Check the lenght

    if length(wigner_distribution.λs) != length(ensemble.rho0.λs)
        error("Different length of the eigenvalues vectors")
    end
 
    if (wigner_distribution.evolve_correlators != ensemble.rho0.evolve_correlators)
        error("The evolution mode is not the same")
    end
    #ensemble.weights .= sqrt(λs' * λs0_inv) # Since we are using the alphas
    
    if (wigner_distribution.evolve_correlators == false)
        ensemble.weights .= prod(sqrt.(wigner_distribution.λs ./ ensemble.rho0.λs))
    else
        ensemble.weights .= prod(sqrt.(ensemble.rho0.λs ./ wigner_distribution.λs))
    end
    #wigner_distribution.λs .= λs 
    #wigner_distribution.λs_vect .= λs_vect

    delta_new = zeros(T, wigner_distribution.n_atoms * 3)
    delta_old = zeros(T, wigner_distribution.n_atoms * 3)
    u_new = zeros(T, length(wigner_distribution.λs))
    u_old = zeros(T, length(wigner_distribution.λs))


    if (wigner_distribution.evolve_correlators == false)
        @views for i = 1:ensemble.n_configs
            delta_new .=  ensemble.positions[:, i] .- wigner_distribution.R_av
            delta_old .=  ensemble.positions[:, i] .- ensemble.rho0.R_av

            u_new .= wigner_distribution.λs_vect' * delta_new 
            u_new .*= sqrt.(wigner_distribution.λs)   # using alpha
            u_old .= (ensemble.rho0.λs_vect' * delta_old) 
            u_old .*= sqrt.(ensemble.rho0.λs)

            numerator = - 0.5 * (u_new' * u_new)
            denominator = - 0.5 *  (u_old' * u_old)
            ensemble.weights[i] *= exp( numerator - denominator)
        end
    else
        @views for i = 1:ensemble.n_configs
            delta_new .=  ensemble.positions[:, i] .- wigner_distribution.R_av
            delta_old .=  ensemble.positions[:, i] .- ensemble.rho0.R_av

            u_new .= wigner_distribution.λs_vect' * delta_new 
            u_new ./= sqrt.(wigner_distribution.λs)   # using alpha
            u_old .= (ensemble.rho0.λs_vect' * delta_old) 
            u_old ./= sqrt.(ensemble.rho0.λs)

            numerator = - 0.5 * (u_new' * u_new)
            denominator = - 0.5 *  (u_old' * u_old)
            ensemble.weights[i] *= exp( numerator - denominator)
        end
    end
end 


function get_average_forces(ensemble :: Ensemble{T}) where {T <: AbstractFloat}

   avg_for = ensemble.forces * ensemble.weights
   #avg_for ./= Float64(ensemble.n_configs)
   #avg_for ./= T(ensemble.n_configs)
   avg_for ./= sum(ensemble.weights)
   println("avg, ", avg_for .* sqrt.(ensemble.rho0.masses) .* CONV_BOHR ./CONV_RY)

end


"""
Evaluate the average dv_dr and d2v_dr2 from the ensemble and the wigner distribution.

The weights on the ensemble should have been already updated.
"""
function get_averages!(dv_dr :: Vector{T}, d2v_dr2 :: Matrix{T}, ensemble :: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T <: AbstractFloat}

    println("forc1")
    println(ensemble.forces[:,1] )
    println("forc sscha")
    println(ensemble.sscha_forces[:,1])
    println("diff forc")
    println(ensemble.forces[:,1] - ensemble.sscha_forces[:,1])
    t0 = time()
    delta = Vector{T}(undef, ensemble.n_configs)
    d2v_dr2_tmp = Matrix{T}(undef, length(wigner_distribution.λs), wigner_distribution.n_modes)

    for i in 1: wigner_distribution.n_atoms * 3
        delta .= ensemble.positions[i,:] .- wigner_distribution.R_av[i]
        delta .*= ensemble.weights
        d2v_dr2[i,:] .=  ensemble.forces  * delta   
    end

    d2v_dr2 ./= sum(ensemble.weights)
    println("<dRdf>")
    display(d2v_dr2 .* sqrt(wigner_distribution.masses[1]) ./ CONV_RY )
    t1 = time()
    println("time = ", t1-t0)

    delta = Vector{T}(undef, ensemble.n_configs)
    d2v_prova  =Matrix{T}(undef, wigner_distribution.n_modes, wigner_distribution.n_modes) 

    for i in 1: wigner_distribution.n_modes
        delta .= ensemble.positions[i,:] .- wigner_distribution.R_av[i]
        for j in 1 : wigner_distribution.n_modes
            d2v_prova[i,j] = sum(ensemble.forces[j,:].* delta .*ensemble.weights)/sum(ensemble.weights)
        end
    end

    t2 = time()
    println("time = ", t2-t1)
    println("diff")
    display(d2v_dr2 .- d2v_prova)

    """
    prova = wigner_distribution.alpha * d2v_dr2 
    lam = eigen(wigner_distribution.alpha)
    ls_vector, ls_values = remove_translations(lam.vectors, lam.values)
    println("check")
    println(ls_values,ones(length(ls_values))./wigner_distribution.λs )
        
    tmp = ls_vector' .* ls_values
    new = zeros(wigner_distribution.n_modes, wigner_distribution.n_modes )
    mul!(new, ls_vector, tmp)
    println("c1")
    println(wigner_distribution.alpha.-new)
    """

    mul!(d2v_dr2_tmp, wigner_distribution.λs_vect' , d2v_dr2)

    d2v_dr2_tmp .= d2v_dr2_tmp ./ wigner_distribution.λs

    mul!(d2v_dr2, wigner_distribution.λs_vect , d2v_dr2_tmp)

    """
    mul!(d2v_dr2_tmp, ls_vector' , d2v_dr2)

    d2v_dr2_tmp .= d2v_dr2_tmp .* ls_values

    mul!(d2v_dr2, ls_vector , d2v_dr2_tmp)
    """


    #d2v_dr2 .+= d2v_dr2'
    #d2v_dr2 ./= 2
    println("fc ")
    display(d2v_dr2 .* wigner_distribution.masses[1])


    """
    println("check ")
    println((d2v_dr2.-prova) .* wigner_distribution.masses[1] ./ CONV_RY .* CONV_BOHR^2 )
    """

    """
    centroid_matrix = repeat(wigner_distribution.R_av,1,ensemble.n_configs)
    d2v_dr2 .= 0

    delta = ensemble.positions .- centroid_matrix
    mul!(d2v_dr2 , delta, ensemble.forces') # sum_{i=1}^{Nconf} delta R_a^i f_c^i

    M = wigner_distribution.alpha - wigner_distribution.gamma' * inv(wigner_distribution.beta) * wigner_distribution.gamma
    d2v_dr2 = -M.*d2v_dr2

    d2v_dr2 .+= d2v_dr2'
    d2v_dr2 ./= 2
    """

    # TODO! Impose the acoustic sum rule and the symmetries
end 

"""
Initialize the ensemble
"""
function init_ensemble_from_python(py_ensemble, settings :: GeneralSettings{T}) where {T <: AbstractFloat}

    # Init the ensemble from python

    dyn0 = py_ensemble.current_dyn
    TEMPERATURE = py_ensemble.T0

    rho0 = QuanumGaussianDynamics.init_from_dyn(dyn0, Float64(TEMPERATURE), settings)
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

    # SSCHA forces
    sscha_forces = reshape(permutedims(py_ensemble.sscha_forces,(3,2,1)), (N_modes, py_ensemble.N))
    sscha_forces .*= CONV_RY ./ CONV_BOHR
    sscha_forces ./= sqrt.(rho0.masses)

    weights = ones( py_ensemble.N)

    ensemble = QuanumGaussianDynamics.Ensemble(rho0 = rho0, positions = ens_positions, forces = ens_forces, n_configs = Int32(py_ensemble.N),
                                          weights = weights, sscha_forces = sscha_forces)

    return ensemble
end


function get_λs(RR_corr :: Matrix{T}) where {T <: AbstractFloat}
    eigvals:: T = eigvals(Hermitian(RR_corr))
    return eigvals
end 


# TODO: add a function to load and save the ensemble on disk
function load_ensemble!(ensemble :: Ensemble{T}, path_to_json :: String) where {T <: AbstractFloat}
end
