using QuantumGaussianDynamics
using DelimitedFiles
using LinearAlgebra
using Unitful, UnitfulAtomic

using PyCall

# Load the python modules of the SSCHA
# To setup the equilibrium state of the H2 molecule
@pyimport cellconstructor as CC 
@pyimport cellconstructor.Phonons as PH
@pyimport sscha
@pyimport sscha.Ensemble as PyEnsemble
@pyimport ase
@pyimport ase.calculators.emt as emt


# Run the calculation
function main()
    method = "generalized-verlet"
    dt = 0.1u"fs"
    t_max = 100.0u"fs"
    temperature = 0.0u"K"


    # Load the SSCHA dynamical matrix
    # @__DIR__ is the directory of the current script
    sscha_path = @__DIR__
    dyn = PH.Phonons(joinpath(sscha_path, "final_result"), 1)

    # Load the converged SSCHA ensemble
    py_ensemble = PyEnsemble.Ensemble(dyn, ustrip(temperature))
    py_ensemble.load_bin(joinpath(sscha_path, "sscha_ensemble"), 1)
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()

    t1 = time()


    """
    Inputs
    dt: discretization time, femtoseconds
    t_max: run time, femtoseconds
    algorithm: evolution algorithm
    kong_liu_ratio: limit of the ensemble stochastic criterion
    evolve_correlators: 
    N: number of configurations
    seed: seed for the calculations. If you do not want to specify any seed, use seed=0
    correlated: whether the dynamics is correlated or not. Avoid with a kong-liu lower than 1. 
    """

    # Properties of the perturbing field
    field_frequency = 1.0u"c" / 500u"nm" # Raman is done with visible light, usually around 500 nm of wavelength. We convert in the frequency with the usual relation νλ = c
    field_duration = 1.0u"fs" # With a duration of 1 fs, we are close to a single cycle pulse.
    field_start_time = 2.0u"fs" # When the field arrives
    field_intensity = 0.0u"V/m" # The intensity of the field.
    field_polarization = [1.0, 0.0, 0.0] # Polarization of the incoming pump (along the molecule)

    # Properties of the simulation
    kong_liu_ratio = 0.2
    N_configs = py_ensemble.N

    settings = QuantumGaussianDynamics.Dynamics(dt, t_max, N_configs; algorithm = method, kong_liu_ratio = kong_liu_ratio, 
                                               verbose = true,  evolve_correlators = true, save_filename = method, 
                                              save_correlators = false, save_each = 1, seed=1254, correlated = false)

    # Prepare the gradient cleaning
    set_clean_gradients!(settings, true)

    rho = QuantumGaussianDynamics.init_from_dyn(dyn, temperature, settings)
    ensemble = QuantumGaussianDynamics.init_ensemble_from_python(py_ensemble, settings)

    # Specify here the ASE calculator
    calculator = emt.EMT()
    crystal = QuantumGaussianDynamics.init_calculator(calculator, rho, ase.Atoms)

    # Apply a Raman tensor for the H2 molecule
    # This info could be stored in the dynamical matrix
    raman_tensor = zeros(Float64, 3, 3, 6)
    for i in 1:3
        raman_tensor[i, i, 1] = 0.01
        raman_tensor[i, i, 4] = -0.01
    end

    # Electric field
    # If you do not want to apply any field, use fake_field, like this. Otherwise, prepare a fake ph.out to read the effective charges and the dielectric constant
    efield = QuantumGaussianDynamics.get_impulsive_raman_pump(raman_tensor,
                                                              field_frequency,
                                                              field_duration,
                                                              field_start_time,
                                                              field_intensity,
                                                              field_polarization)


    # If you want to apply the pulse, specify the parameters
    # Equation of the pulse: E(t)=A*cos(2\pi*freq*t)*exp(-(t-t0)^2/(2*sig^2))
    #A = 3000.0 # 3000.0 #kV/cm
    #freq = 2.4 #THz
    #t0 = 1875.0 #fs
    #sig = 468.0 #fs
    #edir = [0,0,1.0] #Polarization of the field, must have norm=1
    #field_fun = deepcopy(QuantumGaussianDynamics.pulse)
    #field_f = t -> field_fun(t,A,freq,t0,sig)

    # Read effective charges and dielectric tensor from ph.out
    #Zeff, eps = QuantumGaussianDynamics.read_charges_from_out!("ph.out",  rho)
    #efield = QuantumGaussianDynamics.ElectricField(fun = field_f, Zeff = Zeff, edir=edir, eps = eps)

    # Some calculation
    # QuantumGaussianDynamics.generate_ensemble!(settings.N,ensemble, rho)
    # QuantumGaussianDynamics.calculate_ensemble!(ensemble, crystal)

    # Run!
    QuantumGaussianDynamics.integrate!(rho, ensemble, settings, crystal, efield)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end



