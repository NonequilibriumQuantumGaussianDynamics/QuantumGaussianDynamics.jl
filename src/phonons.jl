

function get_alphabeta(TEMP, w_full, pols_full)
    # omega are frequencies in Rydberg
    # T is in Kelvin
    
    if TEMP<0
        error("Temperature must be >= 0, got instead $T")
    end
    
    K_to_Ry=6.336857346553283e-06

    n_mod = length(w_full)
    pols, w = remove_translations(pols_full, w_full, SMALL_VALUE)

    nw = zeros(n_mod - 3)
    aw = zeros(n_mod - 3)
    bw = zeros(n_mod - 3)

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
        cotangent = ones(n_mod - 3)
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


function get_correlators(TEMP, w_full, pols_full)
    # omega are frequencies in Rydberg
    # T is in Kelvin
    
    if TEMP<0
        error("Temperature must be >= 0, got instead $T")
    end
    
    K_to_Ry=6.336857346553283e-06

    n_mod = length(w_full)
    pols, w = remove_translations(pols_full, w_full, SMALL_VALUE)

    nw = zeros(n_mod - 3)
    aw = zeros(n_mod - 3)
    bw = zeros(n_mod - 3)

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
        cotangent = ones(n_mod - 3)
    end
    
    """
    aw .= 2 .* w ./(2.0 .* nw .+ 1)
    bw .= 2 ./(2.0 .* nw .+ 1) ./ w
    """
    aw .= (cotangent) ./ 2 ./ w
    bw .= (cotangent) ./ 2 .* w

    pols_mod = pols .* aw'
    RR_corr = pols * pols_mod'

    pols_mod = pols .* bw'
    PP_corr = pols * pols_mod'
    #return Symmetric(RR_corr), Symmetric(PP_corr)
    RR_corr .= (RR_corr .+ RR_corr')./2.0
    PP_corr .= (PP_corr .+ PP_corr')./2.0
    return RR_corr, PP_corr



    #alpha = zeros(n_mod, n_mod)
    #beta = zeros(n_mod, n_mod)

    
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



function displace_along_mode(mod, eta, wigner, dyn) 
    # eta in Angstrom*sqrt(uma)
    eta = eta * CONV_BOHR * sqrt(CONV_MASS)

    eig, eigv = dyn.DiagonalizeSupercell()
    v = eigv[:,mod]

    du = v .* eta # No need to divide by sqrt(m), by definition of TDSCHA coord

    display(du)
    println("original")
    display(wigner.R_av)
end



    


"""
# TODO: add a function to load and save the ensemble on disk
function load_ensemble!(ensemble :: Ensemble{T}, path_to_json :: String) where {T <: AbstractFloat}
end

"""
