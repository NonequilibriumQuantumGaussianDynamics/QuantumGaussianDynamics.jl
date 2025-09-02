using MPI
using Test
using QuantumGaussianDynamics

using PyCall
using LinearAlgebra
using DelimitedFiles

@pyimport cellconstructor.Phonons as PH
@pyimport cellconstructor as CC
@pyimport  sscha.Ensemble as  PyEnsemble
@pyimport ase
@pyimport ase.calculators.lammpsrun as lammpsrun
@pyimport ase.calculators.lammpslib as lammpslib

MPI.Init()

#Constants to be moved for later use

@testset "Load effective charges and check sum of forces" begin
    TEMPERATURE = 0.0 #FLOAT

    # Load the dyn corresponding to the equilibrium structure of a SSCHA calculation
    sscha_path = "./"
    dyn = PH.Phonons.(sscha_path * "final_dyn", 1)
    py_ensemble = PyEnsemble.Ensemble(dyn, TEMPERATURE)
    py_ensemble.load_bin(sscha_path * "sscha_ensemble", 1)
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()


    # Initialization
    method = "generalized-verlet"
    settings = QuantumGaussianDynamics.Dynamics(dt =0.5, total_time = 10005.0, algorithm = method, kong_liu_ratio =1.0, 
                                               verbose = true,  evolve_correlators = true, save_filename = method, 
                                              save_correlators = true, save_each = 1, N=1000, seed=0, correlated = true)
    rho = QuantumGaussianDynamics.init_from_dyn(dyn, Float64(TEMPERATURE), settings)

    ensemble = QuantumGaussianDynamics.init_ensemble_from_python(py_ensemble, settings)
    QuantumGaussianDynamics.update_weights!(ensemble, rho)

    calc = lammpslib.LAMMPSlib(
            keep_alive=true,
            log_file="log.ase",
            atom_types=Dict("Sr" => 1, "Ti" => 2, "O" => 3),
            lmpcmds=[
                "pair_style flare",
                "pair_coeff * * /n/holyscratch01/kozinsky_lab/libbi/sscha/SrTiO3_flare/srtio3.otf.flare"
                ])

    crystal = QuantumGaussianDynamics.init_calculator(calc, rho, ase.Atoms)

    A = 3000.0 # 3000.0 #kV/cm
    freq = 2.4 #THz
    t0 = 1878.0 #fs
    sig = 468.0 #fs
    edir = [1.0,0.0,0.0]
    field_fun = deepcopy(QuantumGaussianDynamics.pulse)
    field_f = t -> field_fun(t,A,freq,t0,sig)

    Zeff, eps = QuantumGaussianDynamics.read_charges_from_out!("ph.out",  rho)
    efield = QuantumGaussianDynamics.ElectricField(fun = field_f, Zeff = Zeff, edir=edir, eps = eps)

    Nstep = Int32(settings.total_time/settings.dt)
    for i in 1:Nstep
        tim = i*settings.dt
        my_dt = tim/ CONV_FS
        ext_for = QuantumGaussianDynamics.get_external_forces(my_dt, efield, rho)
        ext_for.*=sqrt.(rho.masses)
        for icar=1:3
            fx = ext_for[icar:3:end]
            summ = sum(fx)
            if summ>1e-8
                error("Sum rule broken: ",summ)
            end
        end
    end
    bool=true
    @test bool = true

end

