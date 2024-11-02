
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

function gaussian1(t, A, w, t0)
    A_Ry = A*CONV_EFIELD
    w_Ry = w*CONV_FREQ
    t0_Ry = t0/CONV_FS
    sig_Ry = 1/(2*π*w_Ry)
    

    return -A_Ry * (t-t0_Ry)/sig_Ry * exp(-0.5*(t-t0_Ry)^2/sig_Ry^2 + 0.5)
end

@doc raw"""
    single_cycle_pulse(time :: Real, A :: Quantity, σ :: Quantity, t0 :: Quantity)

The standard single cycle pulse function obtained as the second derivative of a Gaussian function

$$
E(t) = -A \frac{t-t_0}{\sigma^2} \exp\left(-\frac{(t-t_0)^2}{2\sigma^2}\right)
$$

where $A$ is the electric field amplitude (Compatible with V/m units),
σ is the time duration of the pulse (Compatible with fs units) and $t_0$ is the peak intensity of the pulse (Compatible with fs units).
"""
function single_cycle_pulse(t :: Real, A :: Quantity, σ :: Quantity, t0 :: Quantity)
    # Convert to Hartree atomic units and rescale energy to mHa
    A = ustrip(auconvert(A)) * 1000 # Ha to mHa
    σ = ustrip(auconvert(σ)) / 1000
    t0 = ustrip(auconvert(t0)) / 1000

    single_cycle_pulse(t, A, σ, t0)
end
function single_cycle_pulse(t :: Real, A :: Real, σ :: Real, t0 :: Real)
    -A * (t - t0) / σ^2 * exp(-0.5 * (t - t0)^2 / σ^2)
end


function get_external_forces(t::T, efield :: ElectricField{T}, wigner :: WignerDistribution{T}) where {T <: AbstractFloat}

    # t must be in Rydberg units
    # efield.fun must be a function of t only

    efield_norm=norm(efield.edir)
    if abs(efield_norm -1) > SMALL_VALUE
        error("Electric field polarization must have norm 1, found $efield_norm")
    end
    for i in 1:3
        efield_asr_violation = abs(sum(efield.Zeff[:,i])) 
        @views if efield_asr_violation > 1e-4
            error("Must enforce sum rule for effective charges: violated by $efield_asr_violation on component $i")
        end
    end

    @views nat = size(efield.Zeff, 1) ÷ 3
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

function fake_field(nat) :: ElectricField

   Zeff = zeros(3*nat, 3)
   eps = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
   edir = [1.0,0,0]
   field_f = t -> 0

   efield = ElectricField(fun = field_f, Zeff = Zeff, edir=edir, eps = eps)
  
   return efield
  
end


@doc raw"""
    get_IR_electric_field(py_dyn :: PyObject, pol_dir :: AbstractVector, e_function :: Function) :: ElectricField

Return the ElectricField object for the dynamics.
This object describes the light-matter interaction and how the system is driven by the external field in 
the IR regime.
The light-matter coupling of this field occurs through the effective charges and the dielectric constant of the system, i.e. the polarization of the system.

TODO: 
We need to add also the k-vector. This can be used to add the proper LO-TO splitting to the dynamical matrix.

# Arguments

- `py_dyn::PyObject`: The Python cellconstructor Phonons object, containing effective charges and dielectric constant of the system
- `pol_dir::AbstractVector`: The polarization direction of the electric field.
- `e_function::Function`: The function that describes the electric field as a function of time.
"""
function get_IR_electric_field(py_dyn :: PyObject, pol_dir :: AbstractVector{T}, e_function :: Function) :: ElectricField{T} where T
    scell_size = py_dyn.GetSupercell()

    nat = py_dyn.structure.N_atoms
    nat_sc = nat * scell_size[1] * scell_size[2] * scell_size[3] 

    Zeff = zeros(T, 3*nat_sc, 3)
    eps = zeros(T, 3, 3)

    for k in 1:3
        for i in 1:nat_sc
            i_uc = (i - 1) % nat + 1
            for j in 1:3
                Zeff[3*(i-1)+j, k] = py_dyn.effective_charges[i_uc, k, j]
            end
        end
        for j in 1:3
            eps[j, k] = py_dyn.dielectric_tensor[k, j]
        end
    end

    ElectricField(fun = e_function, Zeff = Zeff, edir = pol_dir, eps = eps)
end
