

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

    wigner_distribution.RR_corr .+= (wigner_distribution.RP_corr .+ wigner_distribution.RP_corr')*dt # update RR
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



function generalized_verlet_step!(wigner :: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}, bc0 :: Vector{T} ,part ) where {T <: AbstractFloat}
    # Solve the newton equations
    rank = 0
    size = 1
    if MPI.Initialized()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        size = MPI.Comm_size(MPI.COMM_WORLD)
    end

    if part == 1
        wigner.R_av .+= @. wigner.P_av * dt + 1/2.0 * avg_for * dt^2
        wigner.P_av .+= @. 1/2.0 * avg_for * dt

        tmp_d2v_mul = similar(d2V_dr2)
        println("RR corr ", wigner.RR_corr)
        println("d2V_dr2 ", d2V_dr2)
        mul!(tmp_d2v_mul, wigner.RR_corr, d2V_dr2)  # calculate <RR><d2V>
        println("tmp_d2v_mul ", tmp_d2v_mul)
        wigner.RR_corr .+= @. (wigner.RP_corr + wigner.RP_corr')*dt - (tmp_d2v_mul + tmp_d2v_mul')*dt^2/2.0 + wigner.PP_corr * dt^2
        println("RR corr ", wigner.RR_corr)

        RP_copy = copy(wigner.RP_corr)
        wigner.RP_corr .+= @. 1/2.0 * (wigner.PP_corr - tmp_d2v_mul) *dt
        mul!(tmp_d2v_mul, d2V_dr2, RP_copy)  # calculate <d2V><RP>
        wigner.PP_corr .-= @. 1/2.0 * (tmp_d2v_mul + tmp_d2v_mul') * dt

    elseif part == 2
        wigner.P_av .+= 1/2.0 .* avg_for .* dt # repeat with the new force

        function obj(B1, C1, d2V_dr2, B0p, C0p) 

            KC1 = similar(d2V_dr2)
            mul!(KC1, d2V_dr2, C1)

            dB = @. B0p - 1/2.0 * (KC1 + KC1') *dt
            dC = @. C0p + 1/2.0 * B1 *dt 

            if rank == 0
                println("optimization B ", norm((dB.-B1).^2))
                println("optimization C ", norm((dC.-C1).^2))
            end

            return dB, dC

        end

        B1 = copy(wigner.PP_corr)
        C1 = copy(wigner.RP_corr)

        AK = similar(B1)
        mul!(AK, wigner.RR_corr, d2V_dr2)
        wigner.RP_corr .-= @. 1/2.0 * AK * dt 

        
        niter = 4
        for i in 1 : niter
            B1, C1 = obj(B1, C1, d2V_dr2, wigner.PP_corr, wigner.RP_corr)
        end

        wigner.PP_corr .= B1
        wigner.RP_corr .= C1

        """
        KC = similar(C1)
        mul!(KC, d2V_dr2, C1)  # calculate <d2V><RP>
        AK = similar(C0)
        mul!(AK, wigner.RR_corr, d2V_dr2)
        B0 .-= 1/2.0 * (KC .+ KC') *dt
        C0 .+= 1/2.0 * (B0 .- AK) *dt

        wigner.PP_corr .= B0
        wigner.RP_corr .= C0
        """

    end
end 


function fixed_step!(wigner :: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T},part ) where {T <: AbstractFloat}
    # Solve the newton equations
    if part == 1
        wigner.R_av .+= wigner.P_av .* dt .+ 1/2.0 * avg_for .* dt^2
        wigner.P_av .+= 1/2.0 .* avg_for .* dt

    elseif part == 2
        wigner.P_av .+= 1/2.0 .* avg_for .* dt # repeat with the new force
    end
end 

function full_generalized_verlet_step!(wigner :: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}, bc0 :: Vector{T} ,part ) where {T <: AbstractFloat}
    rank = 0
    size = 1
    if MPI.Initialized()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        size = MPI.Comm_size(MPI.COMM_WORLD)
    end
    # Solve the newton equations
    if part == 1
        wigner.R_av .+= wigner.P_av .* dt .+ 1/2.0 * avg_for .* dt^2
        wigner.P_av .+= 1/2.0 .* avg_for .* dt

        tmp_d2v_mul = similar(d2V_dr2)
        mul!(tmp_d2v_mul, wigner.RR_corr, d2V_dr2)  # calculate <RR><d2V>
        wigner.RR_corr .+= (wigner.RP_corr .+ wigner.RP_corr')*dt .- (tmp_d2v_mul .+ tmp_d2v_mul').*dt^2/2.0 .+ wigner.PP_corr .* dt^2

        RP_copy = copy(wigner.RP_corr)
        wigner.RP_corr .+= 1/2.0 .* (wigner.PP_corr .- tmp_d2v_mul) .*dt
        mul!(tmp_d2v_mul, d2V_dr2, RP_copy)  # calculate <d2V><RP>
        wigner.PP_corr .-= 1/2.0 .* (tmp_d2v_mul .+ tmp_d2v_mul') .* dt

    elseif part == 2
        wigner.P_av .+= 1/2.0 .* avg_for .* dt # repeat with the new force
        
        #KA = similar(d2V_dr2)
        #mul!(KA, d2V_dr2, wigner.RR_corr)  # calculate <RR><d2V>

        function obj(x,p) 
            B0p = p[1]
            C0p = p[2]
            K1 = p[3]
            AK1 = p[4]

            B1, C1 = unmerge_vector(x)
            KC1 = similar(d2V_dr2)
            mul!(KC1, d2V_dr2, C1)

            tot = B1 .- B0p .+ 1/2.0 * (KC1 .+ KC1') *dt
            tot .+= C1 .- C0p .- 1/2.0 * B1 *dt .+ 1/2.0 * AK1 * dt 

            return(norm(tot)^2)
        end

        function grad(x,p) 
            B0p = p[1]
            C0p = p[2]
            K1 = p[3]
            AK1 = p[4]

            B1, C1 = unmerge_vector(x)
            KC1 = similar(d2V_dr2)
            mul!(KC1, d2V_dr2, C1)

            tot = B1 .- B0p .+ 1/2.0 * (KC1 .+ KC1') *dt
            tot .+= C1 .- C0p .- 1/2.0 * B1 *dt .+ 1/2.0 * AK1 * dt 

            gradB = copy(tot)
            gradB  .*= (2+dt)

            gradC = (K1 * tot + tot * K1)*dt + 2 * tot

            """
            println("dt ", dt)
            println("A ", norm(tot))
            println("B ", norm(gradB))
            println("C ", norm(gradC))
            """

            tot_grad =  get_merged_vector(gradB, gradC)
            println("type ", typeof(tot_grad))
            return tot_grad
        end

        B0 = copy(wigner.PP_corr)
        C0 = copy(wigner.RP_corr)
        KC = similar(C0)
        mul!(KC, d2V_dr2, C0)  # calculate <d2V><RP>
        AK = similar(C0)
        mul!(AK, wigner.RR_corr, d2V_dr2)
        B0 .-= 1/2.0 * (KC .+ KC') *dt
        C0 .+= 1/2.0 * (B0 .- AK) *dt
        bc0 = get_merged_vector(B0, C0)
        f = x -> obj(x,  [wigner.PP_corr,  wigner.RP_corr, d2V_dr2, AK])
        g = x -> grad(x,  [wigner.PP_corr,  wigner.RP_corr, d2V_dr2, AK])

        #println("init ", f(bc0))
        if rank==0
          println("init")
        end
        sol = optimize(f, bc0, ConjugateGradient())
        if rank ==0
          println("end")
        end
        bc1 = sol.minimizer
        B0, C0 = unmerge_vector(bc1)
        #println(sol["Status"])
        #error()

        #optf = OptimizationFunction(obj, grad = grad)
        #prob = OptimizationProblem(obj, bc0, [wigner.PP_corr,  wigner.RP_corr, d2V_dr2, AK])
        #sol = solve(optf, GradientDescent())
        #error()
        #println(obj(bc0,  [wigner.PP_corr,  wigner.RP_corr, d2V_dr2, AK]))
        #grad(bc0,  [wigner.PP_corr,  wigner.RP_corr, d2V_dr2, AK])

        wigner.PP_corr .= B0
        wigner.RP_corr .= C0
        #println(sol)
        #error()

        """
        wigner_distribution.RP_corr .+= (wigner_distribution.PP_corr .* dt .- tmp_d2v_mul) # update RP

        mul!(tmp_d2v_mul, d2V_dr2, wigner_distribution.RP_corr) # calculate <d2V><RP>
        tmp_d2v_mul .*= dt

        wigner_distribution.PP_corr .-= (tmp_d2v_mul .+ tmp_d2v_mul')    # update PP

        wigner_distribution.RR_corr .+= (wigner_distribution.RP_corr .+ wigner_distribution.RP_corr') # update RR
        """
    end
end 




function classic_evolution!(Rs::Vector{T}, Ps::Vector{T}, dt::T, cl_for, part) where {T <: AbstractFloat}
        #Ps .+= cl_for .* dt
        #Rs .+= Ps .* dt
        γ = 0.000
        if part == 1
            Rs .+= Ps .* dt + 1/2.0 .* (cl_for .- γ .* Ps) .* dt^2
            Ps .= ((1 - γ/2.0*dt)*Ps .+ 1/2.0 .* cl_for .* dt) ./ (1+γ/2.0*dt)
        elseif part ==2
            Ps .+= (1/2.0 .* cl_for .* dt) ./ (1+γ/2.0*dt)
        end
        #((f + fext + f_old + fext_old)/2.0*dt + (1 - γ/2.0*dt)*u[2])/(1+γ/2.0*dt)
end


function get_merged_vector(BB, CC)

    B = deepcopy(BB)
    C = deepcopy(CC)
    len = size(B)[1]

    b_lin = reshape(B, Int64(len^2))
    c_lin = reshape(C, Int64(len^2))

    return vcat(b_lin,c_lin)
end


function unmerge_vector(x)

    nat3 = Int64(sqrt(length(x)/2.0))
    nat32 = Int64(length(x)/2.0)
    b_lin = x[1:nat32]  
    c_lin = x[nat32+1:end]

    B = reshape(b_lin, (nat3, nat3))
    C = reshape(c_lin, (nat3, nat3))
 
    return B,C
end












