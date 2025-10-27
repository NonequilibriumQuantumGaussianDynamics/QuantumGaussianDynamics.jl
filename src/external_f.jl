
function read_charges_from_dict!(
    dict :: AbstractDict,
    wigner::WignerDistribution{T},
) where {T<:AbstractFloat}
    n_atoms = length(wigner.atoms)
    Zeff = Matrix{T}(undef, 3*n_atoms, 3)
    for i = 1:n_atoms
        start = 3*(i-1)+1
        fin = start + 2
        Zeff[start:fin, :] .= dict[wigner.atoms[i]][:, :]
    end
    return Zeff
end

function read_charges_from_out!(
    filename :: String,
    wigner::WignerDistribution{T},
) where {T<:AbstractFloat}
    n_atoms = length(wigner.atoms)
    Zeff = Matrix{T}(undef, 3*n_atoms, 3)
    eps = Matrix{T}(undef, 3, 3)

    ff = open(filename, "r")
    lines = readlines(ff)
    count = 0
    for i = 1:length(lines)
        line = lines[i]
        if occursin(
            "Effective charges (d Force / dE) in cartesian axis with asr applied",
            line,
        )
            for j = 1:(4*n_atoms)
                line = lines[i+j]
                if occursin("atom", line)
                    count += 1
                    start = 3*(count-1)
                    for k = 1:3
                        l = lines[i+j+k]
                        parts = split(l)
                        Ex = parse(Float64, parts[3])
                        Ey = parse(Float64, parts[4])
                        Ez = parse(Float64, parts[5])
                        Zeff[start+k, :] .= [Ex, Ey, Ez]
                    end
                end
            end
        elseif occursin("Dielectric constant in cartesian axis", line)
            for j = 1:3
                line = lines[i+j+1]
                parts = split(line)
                epsx = parse(Float64, parts[2])
                epsy = parse(Float64, parts[3])
                epsz = parse(Float64, parts[4])
                eps[j, :] .= [epsx, epsy, epsz]
            end
        end
    end


    # sum rule simple
    count = 0.0
    for i = 1:3
        sm = 0
        for j = 1:(3*n_atoms)
            if abs(Zeff[j, i]) > SMALL_VALUE
                count += 1.0
                sm += Zeff[j, i]
            else
                Zeff[j, i] = 0
            end
        end
        sum_Z = sum(Zeff[:, i])
        for j = 1:(3*n_atoms)
            if abs(Zeff[j, i]) > SMALL_VALUE
                Zeff[j, i] -= sum_Z/count
            end
        end
    end
    """
    display(Zeff)
    println("s1 ", sum(Zeff[:,1]))
    println("s2 ", sum(Zeff[:,2]))
    println("s3 ", sum(Zeff[:,3]))
    for i in 1:3
        Zeff[:,i] .-= sum(Zeff[:,i])/(3*n_atoms)
    end
    """

    return Zeff, eps
end

function get_external_forces(
    t::T,
    efield::ElectricField{T},
    wigner::WignerDistribution{T},
) where {T<:AbstractFloat}

    # t must be in Rydberg units
    # efield.fun must be a function of t only

    if abs(norm(efield.edir)-1) > SMALL_VALUE
        error("Electric field direction must have norm 1")
    end
    for i = 1:3
        if abs(sum(efield.Zeff[:, i])) > 1e-4
            error("Must enforce sum rule for effective charges")
        end
    end

    nat = Int32(length(efield.Zeff[:, 1])/3.0)
    forces = Vector{T}(undef, 3*nat)

    for i = 1:nat
        start = 3*(i-1) + 1
        fin = start+2

        epsE = inv(efield.eps)*efield.edir
        ZepsE = efield.Zeff[start:fin, :] * epsE
        forces[start:fin] .=
            ZepsE .* sqrt(2) ./ sqrt(wigner.masses[3*(i-1)+1]) .* efield.fun(t)

    end
    return forces
end

function fake_field(nat::Int32)

    Zeff = zeros(3*nat, 3)
    eps = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    edir = [1.0, 0, 0]
    field_f = t -> 0

    efield = QuantumGaussianDynamics.ElectricField(
        fun = field_f,
        Zeff = Zeff,
        edir = edir,
        eps = eps,
    )

    return efield

end

function fake_dielectric_constant(nat::Int32)

    Zeff = zeros(3*nat, 3)
    eps = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

    return Zeff, eps

end


"""
    sin_field(t, A, w)
Sinusoidal external field. 
"""
function sin_field(A::T, w::T) where {T<:AbstractFloat}
    # t in Rydberg units
    # Amplitude in kV/cm
    # Frequency in THz

    A_Ry = A*CONV_EFIELD
    w_Ry = w*CONV_FREQ

    return (t::Float64) -> A_Ry*sin(2*π*w_Ry*t)
end

"""
    pulse(t, A, w, t0, sig)
Gaussian wavepacket pulse
"""
function pulse(A::T, w::T, t0::T, sig::T) where {T<:AbstractFloat}
    # t in Rydberg units
    # Amplitude in kV/cm
    # Frequency in THz
    # t0 in fs
    # sig in fs

    A_Ry = A*CONV_EFIELD
    w_Ry = w*CONV_FREQ
    t0_Ry = t0/CONV_FS
    sig_Ry = sig/CONV_FS

    return (t::Float64) -> A_Ry * cos(2*π*w_Ry*t) * exp(-0.5*(t-t0_Ry)^2/sig_Ry^2)

end

function gaussian1(A::T, w::T, t0::T) where {T<:AbstractFloat}
    # t in Rydberg units
    # Amplitude in kV/cm
    # Frequency in THz
    # t_0 in fs
    
    A_Ry = A*CONV_EFIELD
    w_Ry = w*CONV_FREQ
    t0_Ry = t0/CONV_FS
    sig_Ry = 1/(2*π*w_Ry)

    return (t::Float64) -> -A_Ry * (t-t0_Ry)/sig_Ry * exp(-0.5*(t-t0_Ry)^2/sig_Ry^2 + 0.5)
end
