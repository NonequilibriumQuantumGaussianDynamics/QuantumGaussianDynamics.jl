t0 = time()

using QuantumGaussianDynamics

using PyCall
using LinearAlgebra
using DelimitedFiles

using Unitful, UnitfulAtomic

@pyimport cellconstructor.Phonons as PH
@pyimport cellconstructor as CC
@pyimport sscha.Ensemble as PyEnsemble
@pyimport ase
@pyimport ase.calculators.emt as emt


function main()
    TEMPERATURE = 0.0u"K"

    # Load the SSCHA dynamical matrix
    sscha_path = @__DIR__
    dyn = PH.Phonons.(joinpath(sscha_path, "final_result"), 1)

    # Load the final SSCHA ensemble
    py_ensemble = PyEnsemble.Ensemble(dyn, TEMPERATURE)
    py_ensemble.load_bin(joinpath(sscha_path, "sscha_ensemble"), 1)
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()

    t1 = time()


    """
    Inputs
    dt: discretization time, femtoseconds
    total_time: run time, femtoseconds
    algorithm: evolution algorithm
    kong_liu_ratio: use 1.0, lower values may not conserve the energy
    evolve_correlators: use true
    N: number of configurations
    seed: seed for the calculations. If you do not want to specify any seed, use seed=0
    correlated: whether the dynamics is correlated or not. 
    """
    #method = "semi-implicit-verlet" # use this one
    method = "generalized-verlet"
    settings = QuantumGaussianDynamics.Dynamics(0.1, 10.0, 100; algorithm = method, kong_liu_ratio = 1.0, 
                                               verbose = true,  evolve_correlators = true, save_filename = method, 
                                              save_correlators = true, save_each = 1, seed=1254, correlated = true)
    rho = QuantumGaussianDynamics.init_from_dyn(dyn, Float64(TEMPERATURE), settings)
    ensemble = QuantumGaussianDynamics.init_ensemble_from_python(py_ensemble, settings)

    # Specify here the ASE calculator
    calculator = emt.EMT()
    crystal = QuantumGaussianDynamics.init_calculator(calculator, rho, ase.Atoms)

    # Electric field
    # If you do not want to apply any field, use fake_field, like this. Otherwise, prepare a fake ph.out to read the effective charges and the dielectric constant
    efield = QuantumGaussianDynamics.fake_field(rho.n_atoms)

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

    # Displacement from equilbrium, optional
    rho.P_av[1] += 0.01 #sqrt(Ha)

    # Some calculation
    QuantumGaussianDynamics.generate_ensemble!(settings.N,ensemble, rho)
    QuantumGaussianDynamics.calculate_ensemble!(ensemble, crystal)

    # Run!
    QuantumGaussianDynamics.integrate!(rho, ensemble, settings, crystal, efield)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end



