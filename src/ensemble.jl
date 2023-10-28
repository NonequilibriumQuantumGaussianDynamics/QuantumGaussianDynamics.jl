
function calculate_ensemble!(ensemble:: Ensemble{T}, crystal) where {T <: AbstractFloat}

    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        sizep = MPI.Comm_size(comm)
        root = 0 
    end


    if MPI.Initialized() == false
        for i in 1 : ensemble.n_configs
            coords  = get_ase_positions(ensemble.positions[:,i] , ensemble.rho0.masses)
            crystal.positions = coords
            energy = crystal.get_potential_energy()
            forces = crystal.get_forces()
            stress = crystal.get_stress()

            forces = reshape(permutedims(forces), Int64(ensemble.rho0.n_modes))
            forces = forces ./ sqrt.(ensemble.rho0.masses) .* CONV_RY ./CONV_BOHR

            ensemble.energies[i] = energy * CONV_RY
            ensemble.forces[:,i] .= forces
            ensemble.stress[:,i] .= stress

            if rank==0
                println("0 Calculating configuration $i out of $(ensemble.n_configs) $(energy * CONV_RY)", )
            elseif rank==1
                println("1 Calculating configuration $i out of $(ensemble.n_configs) $(energy * CONV_RY)")
            end
            
            """
            println("old, ", ensemble.energies[i])
            println("new, ", energy * CONV_RY)

            println("old, ", ensemble.forces[:,i])
            println("new, ", forces)
            """
        end
    else
        start_per_proc, end_per_proc = parallel_force_distribute(ensemble.n_configs)
        nconf_proc = end_per_proc[rank+1] - start_per_proc[rank+1] + 1
        energy_vect = Vector{Float64}(undef,nconf_proc)
        force_array = Matrix{Float64}(undef,Int64(ensemble.rho0.n_modes),nconf_proc)
        stress_array = Matrix{Float64}(undef,6,nconf_proc)

        #for i in 1 : ensemble.n_configs

        for i in start_per_proc[rank+1]: end_per_proc[rank+1]
            if rank == 0
                println("Calculating configuration $i out of $(ensemble.n_configs)")
            end
            coords  = get_ase_positions(ensemble.positions[:,i] , ensemble.rho0.masses)
            crystal.positions = coords
            energy = crystal.get_potential_energy()
            forces = crystal.get_forces()
            stress = crystal.get_stress()

            forces = reshape(permutedims(forces), Int64(ensemble.rho0.n_modes))
            forces = forces ./ sqrt.(ensemble.rho0.masses) .* CONV_RY ./CONV_BOHR

            ind = i - start_per_proc[rank+1] + 1
            energy_vect[ind] = energy * CONV_RY
            force_array[:,ind] .= forces
            stress_array[:,ind] .= stress
            
        end
        energy_length = Vector{Int32}(undef, sizep)
        force_length = Vector{Int32}(undef, sizep)
        stress_length = Vector{Int32}(undef, sizep)
        for i in 1:sizep
            energy_length[i] = end_per_proc[i] - start_per_proc[i] + 1
            force_length[i] = (end_per_proc[i] - start_per_proc[i] + 1)*Int32(ensemble.rho0.n_modes)
            stress_length[i] = (end_per_proc[i] - start_per_proc[i] + 1) *6
        end
        tot_energies = MPI.Allgatherv(energy_vect,energy_length, comm)
        tot_forces = MPI.Allgatherv(force_array,force_length, comm)
        tot_stress = MPI.Allgatherv(stress_array, stress_length, comm)
        tot_forces = reshape(tot_forces,(Int32(ensemble.rho0.n_modes),ensemble.n_configs))
        tot_stress = reshape(tot_stress,(6,ensemble.n_configs))

        ensemble.energies .= tot_energies
        ensemble.forces .= tot_forces
        ensemble.stress .= tot_stress
    end
    """
    check_energy = ensemble.energies .- tot_energies
    check_forces = ensemble.forces .- tot_forces
    println("check ",norm(check_energy))
    println("check for",norm(check_forces))
    MPI.Barrier(comm)
    error()
    """

end

function get_classic_forces(wigner_distribution:: WignerDistribution{T}, crystal) where {T <: AbstractFloat}

        #println("Calculating configuration $i out of $(ensemble.n_configs)")
        coords  = get_ase_positions(wigner_distribution.R_av , wigner_distribution.masses)
        crystal.positions = coords
        energy = crystal.get_potential_energy()
        forces = crystal.get_forces()

        forces = reshape(permutedims(forces), Int64(wigner_distribution.n_modes))
        #println("Classic forces (eV/A)")
        #println(forces)
        forces = forces ./ sqrt.(wigner_distribution.masses) .* CONV_RY ./CONV_BOHR

        #println("Classic forces Ry/Bohr/sqrt(m)")
        #println(forces)
        return forces
end

function get_classic_ef(Rs:: Vector{T}, wigner_distribution:: WignerDistribution{T},  crystal) where {T <: AbstractFloat}

        #println("Calculating configuration $i out of $(ensemble.n_configs)")
        coords  = get_ase_positions(Rs , wigner_distribution.masses)
        crystal.positions = coords
        energy = crystal.get_potential_energy()
        energy *= CONV_RY
        forces = crystal.get_forces()
        forces = reshape(permutedims(forces), Int64(wigner_distribution.n_modes))
        forces = forces ./ sqrt.(wigner_distribution.masses) .* CONV_RY ./CONV_BOHR

        return energy, forces
end

function get_kong_liu(ensemble :: Ensemble{T}) where {T <: AbstractFloat}
    sumrho2 = sum(ensemble.weights.^2)
    sumrho  = sum(ensemble.weights)
    return sumrho^2/sumrho2
end

function generate_ensemble!(N, ensemble:: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T <: AbstractFloat}
    
    ensemble.n_configs = N
    evolve_correlators = wigner_distribution.evolve_correlators
    N_modes = length(wigner_distribution.λs)
    even_odd = true

    if even_odd
        if mod(N,2) !=0 
            error("Error, evenodd allowed only with an even number of random structures")
        end
        N2 = Int64(N/2.0)
        ymu_i = randn(T, (N_modes, N2))
        if MPI.Initialized()
            # So that all the processors work with the same random numbers
            MPI.Bcast!(ymu_i, 0, MPI.COMM_WORLD)
        end
        if evolve_correlators
            ymu_i .*= sqrt.(wigner_distribution.λs) #sqrt(λ)*y_mu
        else
            ymu_i ./= sqrt.(wigner_distribution.λs) #1/sqrt(λ)*y_mu
        end
        dRa_i = wigner_distribution.λs_vect * ymu_i #\sum_{\mu} e_{a\mu}*sqrt(λ_\mu)*y_\mu
        new_positions = Matrix{T}(undef, wigner_distribution.n_modes, N)
        new_positions[:,1:N2] .= dRa_i .+ wigner_distribution.R_av
        new_positions[:,N2+1:end] .= -dRa_i .+ wigner_distribution.R_av
        ensemble.positions = new_positions
    else
        ymu_i = randn(T, (N_modes, N))
        if MPI.Initialized()
            MPI.Bcast!(ymu_i, 0, MPI.COMM_WORLD)
        end
        if evolve_correlators
            ymu_i .*= sqrt.(wigner_distribution.λs) #sqrt(λ)*y_mu
        else
            ymu_i ./= sqrt.(wigner_distribution.λs) #1/sqrt(λ)*y_mu
        end
        dRa_i = wigner_distribution.λs_vect * ymu_i #\sum_{\mu} e_{a\mu}*sqrt(λ_\mu)*y_\mu
        new_positions = Matrix{T}(undef, wigner_distribution.n_atoms, N)
        new_positions .= dRa_i .+ wigner_distribution.R_av
        ensemble.positions = new_positions
    end
    
    # update ensemble
    ensemble.weights = ones(T, N)
    ensemble.rho0 = deepcopy(wigner_distribution)
    ensemble.forces  = zeros(T, (wigner_distribution.n_modes, N))
    ensemble.stress  = zeros(T, (6, N))
    ensemble.sscha_forces = zeros(T, (wigner_distribution.n_modes, N))
    ensemble.energies = zeros(T, N)
    ensemble.sscha_energies = zeros(T, N)

end


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
            #println("delta ", wigner_distribution.R_av .-ensemble.rho0.R_av )

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

function get_total_energy(ensemble :: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T <: AbstractFloat}
   avg_ene = get_average_energy(ensemble)
   K_rho = tr(wigner_distribution.PP_corr)
   K_point = sum(wigner_distribution.P_av.^2 )

   return avg_ene + 0.5*(K_rho + K_point)
end

function get_average_energy(ensemble :: Ensemble{T}) where {T <: AbstractFloat}

   avg_ene = dot(ensemble.energies,ensemble.weights)
   avg_ene /= sum(ensemble.weights)

   std = dot(ensemble.energies .- avg_ene, ensemble.energies .- avg_ene) / (sum(ensemble.weights) -1)
   std = sqrt(std) ./ CONV_RY
   #println("avg ene, ", avg_ene  ./CONV_RY, " std, ", std)
   return avg_ene
end

function get_average_forces(ensemble :: Ensemble{T}) where {T <: AbstractFloat}

   avg_for = ensemble.forces * ensemble.weights
   #avg_for ./= Float64(ensemble.n_configs)
   #avg_for ./= T(ensemble.n_configs)
   avg_for ./= sum(ensemble.weights)
   #println("avg, ")
   #println(avg_for .* sqrt.(ensemble.rho0.masses) )
   #println(avg_for .* sqrt.(ensemble.rho0.masses) .* CONV_BOHR ./CONV_RY)

end


"""
Evaluate the average force and d2v_dr2 from the ensemble and the wigner distribution.

The weights on the ensemble should have been already updated.
"""
function get_averages!(avg_for :: Vector{T}, d2v_dr2 :: Matrix{T}, ensemble :: Ensemble{T}, wigner_distribution :: WignerDistribution{T}) where {T <: AbstractFloat}

    avg_for .= ensemble.forces * ensemble.weights
    avg_for ./= sum(ensemble.weights)

    #t0 = time()
    delta = Vector{T}(undef, ensemble.n_configs)
    d2v_dr2_tmp = Matrix{T}(undef, length(wigner_distribution.λs), wigner_distribution.n_modes)

    for i in 1: wigner_distribution.n_atoms * 3
        delta .= ensemble.positions[i,:] .- wigner_distribution.R_av[i]
        delta .*= ensemble.weights
        d2v_dr2[i,:] .=  (ensemble.forces)  * delta   
    end

    d2v_dr2 ./= sum(ensemble.weights)
    #t1 = time()
    #println("time = ", t1-t0)

    mul!(d2v_dr2_tmp, wigner_distribution.λs_vect' , d2v_dr2)

    d2v_dr2_tmp .= d2v_dr2_tmp ./ wigner_distribution.λs

    mul!(d2v_dr2, wigner_distribution.λs_vect , d2v_dr2_tmp)


    d2v_dr2 .+= d2v_dr2'
    d2v_dr2 ./= (-2.0)
    #println("fc ")
    #display(d2v_dr2 .* wigner_distribution.masses[1])


    # TODO! Impose the acoustic sum rule and the symmetries
end 

"""
Initialize the ensemble
"""
function init_ensemble_from_python(py_ensemble, settings :: Dynamics{T}) where {T <: AbstractFloat}

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

    # Random energies
    ens_energies = py_ensemble.energies #.* CONV_RY

    #SSCHA energies
    sscha_energies = py_ensemble.sscha_energies #.* CONV_RY

    # Random forces
    ens_forces = reshape(permutedims(py_ensemble.forces,(3,2,1)), (N_modes, py_ensemble.N))
    ens_forces = ens_forces ./ CONV_BOHR #.* CONV_RY
    ens_forces = ens_forces ./ sqrt.(rho0.masses)

    # SSCHA forces
    sscha_forces = reshape(permutedims(py_ensemble.sscha_forces,(3,2,1)), (N_modes, py_ensemble.N))
    sscha_forces ./= CONV_BOHR #./ CONV_RY
    sscha_forces ./= sqrt.(rho0.masses)

    # Random stress
    ens_stresses = py_ensemble.stresses
    ens_voigt = Matrix{T}(undef, 6, py_ensemble.N)
    ens_voigt[1,:] = ens_stresses[:,1,1] 
    ens_voigt[2,:] = ens_stresses[:,2,2] 
    ens_voigt[3,:] = ens_stresses[:,3,3] 
    ens_voigt[4,:] = ens_stresses[:,2,3]  #yz
    ens_voigt[5,:] = ens_stresses[:,1,3]  #xz
    ens_voigt[6,:] = ens_stresses[:,1,2]  #xy


    weights = ones( py_ensemble.N)
    if settings.seed != 0 
        Random.seed!(settings.seed)
    end


    ensemble = QuanumGaussianDynamics.Ensemble(rho0 = rho0, positions = ens_positions, forces = ens_forces, stress = ens_voigt,
                                               n_configs = Int32(py_ensemble.N), weights = weights, sscha_forces = sscha_forces,
                                               energies = ens_energies, sscha_energies = sscha_energies,
                                               temperature = TEMPERATURE)

    return ensemble
end


function get_λs(RR_corr :: Matrix{T}) where {T <: AbstractFloat}
    eigvals:: T = eigvals(Hermitian(RR_corr))
    return eigvals
end 


# TODO: add a function to load and save the ensemble on disk
function load_ensemble!(ensemble :: Ensemble{T}, path_to_json :: String) where {T <: AbstractFloat}
end


