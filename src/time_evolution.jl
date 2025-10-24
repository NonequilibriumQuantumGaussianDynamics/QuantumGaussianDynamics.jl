using LinearAlgebra
using Roots
using Optim
using OptimizationOptimJL
using ForwardDiff


"""
    function semi_implicit_verlet_step!(rho:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}, part ) where {T <: AbstractFloat}

Evolves the quantum means and correlators of a `WignerDistribution` by one
semi-implicit Verlet integration step. This algorithms has error O(dt^2) on 
the integration of the correlators. 

This is a two-stage scheme:  
- `part == 1` updates the positions (`R_av`) and momenta (`P_av`) using the
  current force `avg_for`.  
- `part == 2` completes the step with the updated force, and evolves the
  second-order correlators (`RR_corr`, `RP_corr`, `PP_corr`).

# Arguments
- `ρ::WignerDistribution{T}`: system state (means and correlators), mutated in place.
- `dt::T`: time step.
- `avg_for::Vector{T}`: average force vector ⟨F⟩.
- `d2V_dr2::Matrix{T}`: Hessian (second derivatives of the potential).
- `part::Int`: integration stage:

# Method

Given the following variables name
⟨R⟩   = R_av, ⟨P⟩   = P_av, ⟨f⟩   = avg_for, ⟨RR⟩  = RR_corr, ⟨PP⟩  = PP_corr, ⟨RP⟩  = RP_corr, ⟨d²V⟩ = d2V_dr2

- In part == 1:
  - ⟨R⟩ ← ⟨R⟩ + dt·⟨P⟩ + ½·dt²·⟨f⟩
  - ⟨P⟩ ← ⟨P⟩ + ½·dt·⟨f⟩
- In part == 2:
  - ⟨P⟩ ← ⟨P⟩ + ½·dt·⟨f⟩
  - ⟨RP⟩ ← ⟨RP⟩ + dt·⟨PP⟩ − dt·⟨RR⟩·⟨d²V⟩
  - ⟨PP⟩ ← ⟨PP⟩ − dt·(⟨d²V⟩·⟨RP⟩ + (⟨d²V⟩·⟨RP⟩)ᵀ)
  - ⟨RR⟩ ← ⟨RR⟩ + dt·(⟨RP⟩ + ⟨RP⟩ᵀ)

"""
function semi_implicit_verlet_step!(
    rho::WignerDistribution{T},
    dt::T,
    avg_for::Vector{T},
    d2V_dr2::Matrix{T},
    part,
) where {T<:AbstractFloat}

    dthalf = dt/2.0
    dtsq = dt^2/2.0

    if part == 1
        #rho.R_av .+= rho.P_av .* dt .+ 1/2.0 * avg_for .* dt^2
        axpy!(dt, rho.P_av, rho.R_av)
        axpy!(dtsq, avg_for, rho.R_av)
        #rho.P_av .+= 1/2.0 .* avg_for .* dt
        axpy!(dthalf, avg_for, rho.P_av)
    elseif part == 2
        #rho.P_av .+= 1/2.0 .* avg_for .* dt # repeat with the new force
        axpy!(dthalf, avg_for, rho.P_av)

        tmp = similar(d2V_dr2)

        # Evolve the correlators
        #mul!(tmp_d2v_mul, rho.RR_corr, d2V_dr2)  # calculate <RR><d2V>
        #tmp_d2v_mul .*= dt
        #rho.RP_corr .+= (rho.PP_corr .* dt .- tmp_d2v_mul) # update RP
        BLAS.axpy!(dt, rho.PP_corr, rho.RP_corr)
        BLAS.gemm!('N', 'N', -dt, rho.RR_corr, d2V_dr2, 1.0, rho.RP_corr)

        mul!(tmp, d2V_dr2, rho.RP_corr) # calculate <d2V><RP>
        #tmp .*= dt
        #rho.PP_corr .-= (tmp .+ tmp')    # update PP
        axpy!(-dt, tmp, rho.PP_corr)
        axpy!(-dt, tmp', rho.PP_corr)

        #rho.RR_corr .+= (rho.RP_corr .+ rho.RP_corr')*dt # update RR
        axpy!(dt, rho.RP_corr, rho.RR_corr)
        axpy!(dt, rho.RP_corr', rho.RR_corr)
    end
end

"""

    generalized_verlet_step!(rho :: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}, bc0 :: Vector{T} ,part ) where {T <: AbstractFloat}

Evolves the means and second–order correlators of a `WignerDistribution` by one
**generalized-Verlet** step. This is algorithm has O(dt^3) error in time step. 
The algorithnm is illustrated in the work from Libbi et al., npj Computat. Mater. 11, 102 (2025), and
consists of the explicit evolution of the variables ⟨R⟩, ⟨P⟩ and ⟨RR⟩, and of an implicit evolution for
the variables ⟨PP⟩ and ⟨RP⟩. The latter is performed iteratively. 

# Arguments
- `rho::WignerDistribution{T}`: state (means & correlators), updated in place.
- `dt::T`: time step.
- `avg_for::Vector{T}`: current average force ⟨f⟩.
- `d2V_dr2::Matrix{T}`: Hessian (∂²V). Assumed symmetric in practice.
- `part::Int`: stage selector (`1` = predictor, `2` = corrector).

Given the following variables name
⟨R⟩   = R_av, ⟨P⟩   = P_av, ⟨f⟩   = avg_for, ⟨RR⟩  = RR_corr, ⟨PP⟩  = PP_corr, ⟨RP⟩  = RP_corr, ⟨d²V⟩ = d2V_dr2

- `part == 1` (predictor):
  - Means (half-kick)
    - ⟨R⟩ ← ⟨R⟩ + dt*⟨P⟩ + ½*dt²*⟨f⟩
    - ⟨P⟩ ← ⟨P⟩ + ½*dt*⟨f⟩
  - Correlators
    - let  K = ⟨RR⟩ * ⟨d²V⟩
    - ⟨RR⟩ ← ⟨RR⟩ + dt*(⟨RP⟩ + ⟨RP⟩ᵀ) − ½*dt²*(K + Kᵀ) + dt²*⟨PP⟩
    - save ⟨RP⟩₀ = ⟨RP⟩
    - ⟨RP⟩ ← ⟨RP⟩ + ½*dt*(⟨PP⟩ − K)
    - ⟨PP⟩ ← ⟨PP⟩ − ½*dt*(⟨d²V⟩*⟨RP⟩₀ + (⟨d²V⟩*⟨RP⟩₀)ᵀ)

- `part == 2` (corrector / finishing kick):
  - Means
    - ⟨P⟩ ← ⟨P⟩ + ½*dt*⟨f⟩
  - Correlators (fixed-point refinement of ⟨PP⟩ and ⟨RP⟩)
    - pre-update: ⟨RP⟩ ← ⟨RP⟩ − ½*dt*(⟨RR⟩*⟨d²V⟩)
    - iterate a few times (typically 4):
      - ⟨RP⟩₁ ← ⟨RP⟩ + ½*dt*⟨PP⟩₁
      - ⟨PP⟩₁ ← ⟨PP⟩ − ½*dt*(⟨d²V⟩*⟨RP⟩₁ + (⟨d²V⟩*⟨RP⟩₁)ᵀ)
    - write back final ⟨PP⟩, ⟨RP⟩

"""
function generalized_verlet_step!(
    rho::WignerDistribution{T},
    dt::T,
    avg_for::Vector{T},
    d2V_dr2::Matrix{T},
    part,
) where {T<:AbstractFloat}

    if part == 1
        @. rho.R_av += rho.P_av * dt + 1/2.0 * avg_for * dt^2
        @. rho.P_av += 1/2.0 * avg_for * dt

        tmp_d2v_mul = similar(d2V_dr2)
        mul!(tmp_d2v_mul, rho.RR_corr, d2V_dr2)  # calculate <RR><d2V>
        @. rho.RR_corr +=
            (rho.RP_corr + rho.RP_corr')*dt - (tmp_d2v_mul + tmp_d2v_mul')*dt^2/2.0 +
            rho.PP_corr * dt^2

        RP_copy = copy(rho.RP_corr)
        @. rho.RP_corr += 1/2.0 * (rho.PP_corr - tmp_d2v_mul) * dt
        mul!(tmp_d2v_mul, d2V_dr2, RP_copy)  # calculate <d2V><RP>
        @. rho.PP_corr -= 1/2.0 * (tmp_d2v_mul + tmp_d2v_mul') * dt

    elseif part == 2
        @. rho.P_av += 1/2.0 * avg_for * dt # repeat with the new force

        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        function obj(B1, C1, d2V_dr2, B0p, C0p)

            C1_ = @. C0p + 1/2.0 * B1 * dt

            B1_ = copy(B0p)
            BLAS.syr2k!('U', 'T', -dt/2.0, d2V_dr2, C1, 1.0, B1_)
            LinearAlgebra.copytri!(B1_, 'U')

            if rank == 0
                println("optimization B ", norm((B1_ .- B1) .^ 2))
                println("optimization C ", norm((C1_ .- C1) .^ 2))
            end

            return B1_, C1_

        end

        B1 = copy(rho.PP_corr)
        C1 = copy(rho.RP_corr)

        dthalf = dt / 2.0
        BLAS.gemm!('N', 'N', -dthalf, rho.RR_corr, d2V_dr2, 1.0, rho.RP_corr)

        niter = 4
        for i = 1:niter
            B1, C1 = obj(B1, C1, d2V_dr2, rho.PP_corr, rho.RP_corr)
        end

        rho.PP_corr .= B1
        rho.RP_corr .= C1

    end
end

"""
   function fixed_step!(wigner :: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T},part ) where {T <: AbstractFloat}

Evolves the quantum state freezing the nuclear degrees of freedom. This corresponds to a classic evolution 
in which however the forces are computed as ensemble averages using the initial quantum distribution.
"""
function fixed_step!(
    wigner::WignerDistribution{T},
    dt::T,
    avg_for::Vector{T},
    d2V_dr2::Matrix{T},
    part,
) where {T<:AbstractFloat}

    dthalf = dt/2.0
    dtsq = dt^2/2.0

    if part == 1
        axpy!(dt, rho.P_av, rho.R_av)
        axpy!(dtsq, avg_for, rho.R_av)
        axpy!(dthalf, avg_for, rho.P_av)

    elseif part == 2
        axpy!(dthalf, avg_for, rho.P_av)
    end
end


"""
   classic_evolution!(Rs::Vector{T}, Ps::Vector{T}, dt::T, cl_for, part) where {T <: AbstractFloat}

Classic dynamics.
"""
function classic_evolution!(
    Rs::Vector{T},
    Ps::Vector{T},
    dt::T,
    cl_for,
    part,
) where {T<:AbstractFloat}

    γ = 0.000
    if part == 1
        @. Rs += Ps * dt + 1/2.0 * (cl_for - γ * Ps) * dt^2
        @. Ps = ((1 - γ/2.0*dt)*Ps + 1/2.0 * cl_for * dt) / (1+γ/2.0*dt)
    elseif part == 2
        @. Ps += (1/2.0 * cl_for * dt) ./ (1+γ/2.0*dt)
    end

end

"""
   euler_step!(wigner:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}) where {T <: AbstractFloat}

Explicit Euler method. Implemented for testing and didactic use. MUST BE AVOIDED, it is unconditionally unstable. 
"""
function euler_step!(
    wigner::WignerDistribution{T},
    dt::T,
    avg_for::Vector{T},
    d2V_dr2::Matrix{T},
) where {T<:AbstractFloat}
    # Solve the newton equations
    @. wigner.P_av += avg_for * dt
    @. wigner.R_av += wigner.P_av * dt

    tmp_d2v_mul = similar(d2V_dr2)
    copy_RP_corr = copy(wigner.RP_corr)
    copy_RP_corr .*= dt
    copy_PP_corr = copy(wigner.PP_corr) # Debug


    # Evolve the correlators
    mul!(tmp_d2v_mul, d2V_dr2, wigner.RP_corr) # first calculate <d2V><RP>
    tmp_d2v_mul .*= dt

    @. wigner_distribution.RP_corr += wigner.PP_corr * dt # update RP

    @. wigner_distribution.PP_corr -= (tmp_d2v_mul + tmp_d2v_mul')    # update PP

    mul!(tmp_d2v_mul, wigner.RR_corr, d2V_dr2)
    tmp_d2v_mul .*= dt
    wigner.RP_corr .-= tmp_d2v_mul      # update RP

    wigner.RR_corr .+= copy_RP_corr
    wigner.RR_corr .+= copy_RP_corr'
end

"""
   semi_implicit_euler_step!(wigner:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}) where {T <: AbstractFloat}

Semi-implicit Euler method. Implemented for testing and didactic use. Not efficient, use semi-implicit Verlet instead.
"""
function semi_implicit_euler_step!(
    wigner::WignerDistribution{T},
    dt::T,
    avg_for::Vector{T},
    d2V_dr2::Matrix{T},
) where {T<:AbstractFloat}
    # Solve the newton equations
    @. wigner.P_av += avg_for * dt
    @. wigner.R_av += wigner.P_av * dt

    tmp_d2v_mul = similar(d2V_dr2)
    copy_RP_corr = copy(wigner.RP_corr)
    copy_RP_corr .*= dt
    copy_PP_corr = copy(wigner.PP_corr) # Debug


    # Evolve the correlators
    mul!(tmp_d2v_mul, wigner.RR_corr, d2V_dr2)  # calculate <RR><d2V>
    tmp_d2v_mul .*= dt

    @. wigner.RP_corr += (wigner.PP_corr * dt - tmp_d2v_mul) # update RP

    mul!(tmp_d2v_mul, d2V_dr2, wigner.RP_corr) # calculate <d2V><RP>
    tmp_d2v_mul .*= dt

    @. wigner.PP_corr -= (tmp_d2v_mul + tmp_d2v_mul')    # update PP

    @. wigner.RR_corr += (wigner.RP_corr + wigner.RP_corr')*dt # update RR
end



"""
   semi_implicit_euler_step!(wigner:: WignerDistribution{T}, dt :: T, avg_for :: Vector{T}, d2V_dr2 :: Matrix{T}) where {T <: AbstractFloat}

Experimental.
"""
function full_generalized_verlet_step!(
    wigner::WignerDistribution{T},
    dt::T,
    avg_for::Vector{T},
    d2V_dr2::Matrix{T},
    bc0::Vector{T},
    part,
) where {T<:AbstractFloat}
    # Solve the newton equations
    if part == 1
        wigner.R_av .+= wigner.P_av .* dt .+ 1/2.0 * avg_for .* dt^2
        wigner.P_av .+= 1/2.0 .* avg_for .* dt

        tmp_d2v_mul = similar(d2V_dr2)
        mul!(tmp_d2v_mul, wigner.RR_corr, d2V_dr2)  # calculate <RR><d2V>
        wigner.RR_corr .+=
            (wigner.RP_corr .+ wigner.RP_corr')*dt .-
            (tmp_d2v_mul .+ tmp_d2v_mul') .* dt^2/2.0 .+ wigner.PP_corr .* dt^2

        RP_copy = copy(wigner.RP_corr)
        wigner.RP_corr .+= 1/2.0 .* (wigner.PP_corr .- tmp_d2v_mul) .* dt
        mul!(tmp_d2v_mul, d2V_dr2, RP_copy)  # calculate <d2V><RP>
        wigner.PP_corr .-= 1/2.0 .* (tmp_d2v_mul .+ tmp_d2v_mul') .* dt

    elseif part == 2
        wigner.P_av .+= 1/2.0 .* avg_for .* dt # repeat with the new force

        #KA = similar(d2V_dr2)
        #mul!(KA, d2V_dr2, wigner.RR_corr)  # calculate <RR><d2V>

        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        function obj(x, p)
            B0p = p[1]
            C0p = p[2]
            K1 = p[3]
            AK1 = p[4]

            B1, C1 = unmerge_vector(x)
            KC1 = similar(d2V_dr2)
            mul!(KC1, d2V_dr2, C1)

            tot = B1 .- B0p .+ 1/2.0 * (KC1 .+ KC1') * dt
            tot .+= C1 .- C0p .- 1/2.0 * B1 * dt .+ 1/2.0 * AK1 * dt

            return (norm(tot)^2)
        end

        function grad(x, p)
            B0p = p[1]
            C0p = p[2]
            K1 = p[3]
            AK1 = p[4]

            B1, C1 = unmerge_vector(x)
            KC1 = similar(d2V_dr2)
            mul!(KC1, d2V_dr2, C1)

            tot = B1 .- B0p .+ 1/2.0 * (KC1 .+ KC1') * dt
            tot .+= C1 .- C0p .- 1/2.0 * B1 * dt .+ 1/2.0 * AK1 * dt

            gradB = copy(tot)
            gradB .*= (2+dt)

            gradC = (K1 * tot + tot * K1)*dt + 2 * tot

            """
            println("dt ", dt)
            println("A ", norm(tot))
            println("B ", norm(gradB))
            println("C ", norm(gradC))
            """

            tot_grad = get_merged_vector(gradB, gradC)
            println("type ", typeof(tot_grad))
            return tot_grad
        end

        B0 = copy(wigner.PP_corr)
        C0 = copy(wigner.RP_corr)
        KC = similar(C0)
        mul!(KC, d2V_dr2, C0)  # calculate <d2V><RP>
        AK = similar(C0)
        mul!(AK, wigner.RR_corr, d2V_dr2)
        B0 .-= 1/2.0 * (KC .+ KC') * dt
        C0 .+= 1/2.0 * (B0 .- AK) * dt
        bc0 = get_merged_vector(B0, C0)
        f = x -> obj(x, [wigner.PP_corr, wigner.RP_corr, d2V_dr2, AK])
        g = x -> grad(x, [wigner.PP_corr, wigner.RP_corr, d2V_dr2, AK])

        #println("init ", f(bc0))
        if rank==0
            println("init")
        end
        sol = optimize(f, bc0, ConjugateGradient())
        if rank == 0
            println("end")
        end
        bc1 = sol.minimizer
        B0, C0 = unmerge_vector(bc1)

        #optf = OptimizationFunction(obj, grad = grad)
        #prob = OptimizationProblem(obj, bc0, [wigner.PP_corr,  wigner.RP_corr, d2V_dr2, AK])
        #sol = solve(optf, GradientDescent())
        #error()
        #println(obj(bc0,  [wigner.PP_corr,  wigner.RP_corr, d2V_dr2, AK]))
        #grad(bc0,  [wigner.PP_corr,  wigner.RP_corr, d2V_dr2, AK])

        wigner.PP_corr .= B0
        wigner.RP_corr .= C0
    end
end



function get_merged_vector(BB, CC)

    B = deepcopy(BB)
    C = deepcopy(CC)
    len = size(B)[1]

    b_lin = reshape(B, Int64(len^2))
    c_lin = reshape(C, Int64(len^2))

    return vcat(b_lin, c_lin)
end


function unmerge_vector(x)

    nat3 = Int64(sqrt(length(x)/2.0))
    nat32 = Int64(length(x)/2.0)
    b_lin = x[1:nat32]
    c_lin = x[(nat32+1):end]

    B = reshape(b_lin, (nat3, nat3))
    C = reshape(c_lin, (nat3, nat3))

    return B, C
end
