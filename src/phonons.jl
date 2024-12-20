

@doc raw"""
    get_alphabeta(TEMP, w_full, pols_full, settings :: GeneralSettings)


Return the α and β parameters from the frequencies
"""
function get_alphabeta(TEMP, w_full, pols_full, settings :: GeneralSettings)
    # omega are frequencies in Rydberg
    # T is in Kelvin
    
    if TEMP<0
        error("Temperature must be >= 0, got instead $T")
    end
    
    K_to_Ry=6.336857346553283e-06

    n_mod = length(w_full)
    pols, w = remove_translations(pols_full, w_full, settings :: GeneralSettings)
    
    n_translations = get_n_translations(w_full,settings) ## in 
    nw = zeros(n_mod - n_translations)
    aw = zeros(n_mod - n_translations)
    bw = zeros(n_mod - n_translations)

    if TEMP > SMALL_VALUE
        """
        arg = w./(TEMP.*K_to_Ry)
        arg[arg .> 10.0] .= 10.0
        nw .= 1 ./(exp.(arg).-1)
        nw[nw .< SMALL_VALUE] .= 0.0
        """
        arg = w./(TEMP.*K_to_Ry) ./ 2.0
        cotangent = coth.(arg)
    else
        cotangent = ones(n_mod - n_translations)
    end
    
    """
    aw .= 2 .* w ./(2.0 .* nw .+ 1)
    bw .= 2 ./(2.0 .* nw .+ 1) ./ w
    """
    aw .= 2 .* w ./(cotangent)
    bw .= 2 ./(cotangent) ./  w

    pols_mod = pols .* aw'
    alpha = pols * pols_mod'

    pols_mod = pols .* bw'
    beta = pols * pols_mod'
    #return Symmetric(alpha), Symmetric(beta)
    alpha .= (alpha .+ alpha')./2.0
    beta .= (beta .+ beta')./2.0
    return alpha, beta

    #alpha = zeros(n_mod, n_mod)
    #beta = zeros(n_mod, n_mod)

    
end


@doc raw"""
    get_correlators(TEMP, w_full, pols_full, settings :: GeneralSettings)

Convert the frequencies and polarization vector into the
mass-rescaled correlator matrices.

# Arguments
- `TEMP`: Temperature in Kelvin (or Unitful Quantity, internally converted in Kelvin)
- `w_full`: Pulsations in hartree atomic units (ħω)
- `pols_full`: Polarization vectors (unitless)
- `settings`: GeneralSettings object

# Returns
- `RR_corr`: Mass-rescaled RR correlator matrix
- `PP_corr`: Mass-rescaled PP correlator matrix
"""
function get_correlators(TEMP, w_full :: AbstractVector{T}, pols_full :: AbstractMatrix{T}, settings :: GeneralSettings) where {T}
    if TEMP<0
        error("Temperature must be >= 0, got instead $T")
    end
    
    # K_to_Ry=6.336857346553283e-06
    kT = TEMP * CONV_K

    n_mod = length(w_full)
    pols, w = remove_translations(pols_full, w_full, settings)

    n_translations = get_n_translations(w_full, settings)
    nw = zeros(n_mod - n_translations)
    aw = zeros(n_mod - n_translations)
    bw = zeros(n_mod - n_translations)

    if TEMP > SMALL_VALUE
        """
        arg = w./(TEMP.*K_to_Ry)
        arg[arg .> 10.0] .= 10.0
        nw .= 1 ./(exp.(arg).-1)
        nw[nw .< SMALL_VALUE] .= 0.0
        """
        arg = w./kT ./ 2.0
        cotangent = coth.(arg)
    else
        cotangent = ones(n_mod - n_translations)
    end
    
    """
    aw .= 2 .* w ./(2.0 .* nw .+ 1)
    bw .= 2 ./(2.0 .* nw .+ 1) ./ w
    """
    aw .= (cotangent) ./ (2w)
    bw .= (cotangent) ./ 2 .* w

    pols_mod = pols .* aw'
    RR_corr = pols * pols_mod'

    pols_mod = pols .* bw'
    PP_corr = pols * pols_mod'
    #return Symmetric(RR_corr), Symmetric(PP_corr)
    RR_corr .= (RR_corr .+ RR_corr')./2.0
    PP_corr .= (PP_corr .+ PP_corr')./2.0
    return RR_corr, PP_corr
end
function get_correlators(temperature :: Quantity, w_full :: AbstractVector{T}, pols_full :: AbstractMatrix{T}, settings :: GeneralSettings) where {T}
    return get_correlators(ustrip(uconvert(u"K", temperature)), w_full, pols_full, settings)
end



function extract_dynamical_matrix(wigner :: WignerDistribution{T}, TEMP) where {T <: AbstractFloat}
        
    function psi(w,TEMP,val)
        K_to_Ry=6.336857346553283e-06
        arg = w./(TEMP.*K_to_Ry) ./ 2.0
        cotangent = coth.(arg)
        return cotangent / (2.0*w) - val
    end


    nmodes = length(wigner.λs)
    for i in 1: nmodes
        lambda = wigner.λs[i]

        xmin = 1/(2.0*lambda)
        xmax = sqrt(2/lambda)

        psix = x -> psi(x,TEMP,lambda)
        omeg = find_zeros(psix, (xmin,xmax))
        if length(omeg) == 0 
            error("Root finding failed")
        end
    end 

end



function displace_along_mode!(mod, eta, wigner, dyn) 
    # eta in Angstrom*sqrt(uma)
    eta = eta * CONV_BOHR * sqrt(CONV_MASS)

    eig, eigv = dyn.DiagonalizeSupercell()
    v = eigv[:,mod]

    du = v .* eta # No need to divide by sqrt(m), by definition of TDSCHA coord
    wigner.R_av .+= du
end


function update!(wigner :: WignerDistribution, settings :: GeneralSettings)
    # Override the number of dimensions
    if settings isa ASR
        settings.n_dims = get_ndims(wigner)
    end

    # println("[UPDATE] RR corr = ", wigner.RR_corr)
    lambda_eigen = eigen((wigner.RR_corr))
    println(" DEBUG λs = ", lambda_eigen.values)
    λvects, λs = remove_translations(lambda_eigen.vectors, lambda_eigen.values, settings)
    wigner.λs_vect = λvects
    wigner.λs = λs
    println(" After translations = ", λs)

    # println("[UPDATE] λs = ", wigner.λs)
    # println("[UPDATE] λs_vect = ", wigner.λs_vect)
end
function update!(wigner :: WignerDistribution, settings :: Dynamics)
    update!(wigner, get_general_settings(settings))
end


@doc raw"""
    get_ω(λ :: T, temperature :: U) :: T where {T, U}

Returns the value of the auxiliary frequency from the eigenvalue of the
RR correlator matrix.
It is given by the formula:

$$
\lambda = \frac{2n(\omega) + 1}{2\omega}
$$
where $n(\omega)$ is the Bose-Einstein distribution function.
"""
function get_ω(λ :: T, temperature :: T; thr = 1e-5) :: T where {T}
    if temperature < SMALL_VALUE
        return 1.0 / (2.0 * λ)
    end

    kT = temperature * CONV_K
    n_occ(ω) = 1/(exp(ω/kT) - 1)
    get_λ(ω) = (2n_occ(ω) + 1)/(2ω)

    # Start value, let us use classical limit with a small quantum correction
    ω = 1 + √(1 + 16kT * λ)
    ω /= 4λ

    # Newton-Raphson
    δ = Inf
    count = 0
    while abs(δ) > thr && count < 100
        δ = get_λ(ω) - λ
        diff = ForwardDiff.derivative(get_λ, ω)
        ω -= δ / diff
        count += 1
    end
    
    ω
end

@doc raw"""
    get_Φ!(Φ :: AbstractMatrix{T}, λs :: AbstractVector{T}, λ_pols :: AbstractMatrix{T}, temperature ::T) where {T}
    get_Φ!(Φ :: AbstractMatrix{T}, wigner :: WignerDistribution{T}, temperature ::T) where {T}
    get_Φ(wigner :: WignerDistribution{T}, temperature ::T) where {T}

Get the effective force constant matrix Φ from the eigenvalues of the RR correlator matrix.
We only use the eigenvalues and eigenvectors of the RR correlators.
"""
function get_Φ!(Φ :: AbstractMatrix{T}, λs :: AbstractVector{T}, λ_pols :: AbstractMatrix{T}, temperature ::T) where {T}
    n_modes = size(Φ, 1)
    n_good_modes = length(λs)

    # println("SIZE Φ: ", size(Φ))
    # println("SIZE λs: ", size(λs))
    # println("SIZE λ_pols: ", size(λ_pols))
    
    Φ .= 0.0
    for μ in 1:n_good_modes
        ω_μ = get_ω(λs[μ], temperature)
        @views Φ .+= λ_pols[:, μ] * λ_pols[:, μ]' * ω_μ^2
    end
end
function get_Φ!(Φ :: AbstractMatrix{T}, wigner :: WignerDistribution{T}, temperature ::T) where {T}
    get_Φ!(Φ, wigner.λs, wigner.λs_vect, temperature)
end
function get_Φ(wigner :: WignerDistribution{T}, temperature ::T) where {T}
    nadims = get_nmodes(wigner)
    Φ = zeros(T, nadims, nadims)
    get_Φ!(Φ, wigner, temperature)
    return Φ
end


function get_volume(wigner :: WignerDistribution{T}) where {T}
    return abs(det(get_cell(wigner)))
end

    


"""
# TODO: add a function to load and save the ensemble on disk
function load_ensemble!(ensemble :: Ensemble{T}, path_to_json :: String) where {T <: AbstractFloat}
end

"""
