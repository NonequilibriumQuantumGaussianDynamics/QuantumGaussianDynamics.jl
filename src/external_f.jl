
function read_charges_from_dict!(dict,  wigner:: WignerDistribution{T}) where {T <: AbstractFloat}
   n_atoms = length(wigner.atoms)
   Zeff = Matrix{T}(undef, 3*n_atoms, 3)
   for i in 1:n_atoms
       start = 3*(i-1)+1
       fin = start + 2
       Zeff[start:fin,:] .= dict[wigner.atoms[i]][:,:]
   end
   return Zeff
end

function read_charges_from_out!(filename,  wigner:: WignerDistribution{T}) where {T <: AbstractFloat}
   n_atoms = length(wigner.atoms)
   Zeff = Matrix{T}(undef, 3*n_atoms, 3)
   eps = Matrix{T}(undef, 3,3)

   ff = open(filename, "r")
   lines = readlines(ff)
   count = 0
   for i in 1:length(lines)
       line = lines[i]
       if occursin("Effective charges (d Force / dE) in cartesian axis with asr applied" , line)
           for j = 1:4*n_atoms
               line = lines[i+j]
               if occursin("atom", line)
                   count += 1
                   start = 3*(count-1)
                   for k in 1:3
                       l = lines[i+j+k]
                       parts = split(l)
                       Ex =parse(Float64,parts[3])
                       Ey =parse(Float64,parts[4])
                       Ez =parse(Float64,parts[5])
                       Zeff[start+k,:] .= [Ex,Ey,Ez]
                   end
               end
           end
       elseif occursin("Dielectric constant in cartesian axis",line) 
           for j in 1:3
               line = lines[i+j+1]
               parts = split(line)
               epsx =parse(Float64,parts[2])
               epsy =parse(Float64,parts[3])
               epsz =parse(Float64,parts[4])
               eps[j,:] .=[epsx,epsy,epsz]
           end
       end
    end


   # sum rule simple
   count = 0.0
   for i in 1:3
       sm = 0 
       for j in 1:3*n_atoms
           if abs(Zeff[j,i]) > SMALL_VALUE
               count += 1.0
               sm += Zeff[j,i]
           else
               Zeff[j,i] = 0 
           end
       end
       sum_Z = sum(Zeff[:,i])
       for j in 1:3*n_atoms
           if abs(Zeff[j,i]) > SMALL_VALUE
                 Zeff[j,i] -= sum_Z/count
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

function sin_field(t,A,w)
    # t in Rydberg units
    # Amplitude in kV/cm
    # Frequency in THz

    A_Ry = A*CONV_EFIELD
    w_Ry = w*CONV_FREQ

    return A_Ry*sin(2*π*w_Ry*t)
end

function pulse(t, A, w, t0, sig )

    A_Ry = A*CONV_EFIELD
    w_Ry = w*CONV_FREQ
    t0_Ry = t0/CONV_FS
    sig_Ry = sig/CONV_FS

    return A_Ry * cos(2*π*w_Ry*t) * exp(-0.5*(t-t0_Ry)^2/sig_Ry^2)

end


function get_external_forces(t::T, efield :: ElectricField{T}, wigner :: WignerDistribution{T}) where {T <: AbstractFloat}

    # t must be in Rydberg units
    # efield.fun must be a function of t only

    if abs(norm(efield.edir)-1) > SMALL_VALUE
        error("Electric field direction must have norm 1")
    end
    for i in 1:3
        if abs(sum(efield.Zeff[:,i])) > 1e-4
            error("Must enforce sum rule for effective charges")
        end
    end

    nat = Int32(length(efield.Zeff[:,1])/3.0)
    forces = Vector{T}(undef, 3*nat)

    for i in 1:nat
        start = 3*(i-1) +1
        fin = start+2

        epsE = inv(efield.eps)*efield.edir
        ZepsE  = efield.Zeff[start:fin,:]  * epsE
        forces[start:fin] .= ZepsE .* sqrt(2) ./sqrt(wigner.masses[3*(i-1)+1]).* efield.fun(t)

    end
    return forces
end


