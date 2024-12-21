    

function Dynamics(dt:: Quantity, 
        total_time:: Quantity, 
        N :: Int;
        algorithm:: String = "generalized-verlet",
        kong_liu_ratio:: AbstractFloat = 1.0,
        verbose:: Bool = true,
        evolve_correlators:: Bool = true,
        seed:: Int = 0,
        evolve_correlated:: Bool = true,
        settings:: GeneralSettings = ASR(; n_dims=3),
        save_filename:: String = "dynamics",
        save_correlators:: Bool = false,
        save_each:: Int=100)

    return Dynamics(ustrip(uconvert(u"fs", dt)), 
                    ustrip(uconvert(u"fs", total_time)), 
                    N;
                    algorithm=algorithm, 
                    kong_liu_ratio=kong_liu_ratio, 
                    verbose=verbose, 
                    evolve_correlators=evolve_correlators, 
                    settings=settings,
                    seed=seed, 
                    save_filename=save_filename,
                    save_correlators=save_correlators,
                    save_each=save_each)
end


function init_from_dyn(dyn :: PyObject, temperature :: Quantity, settings :: Dynamics)
    return init_from_dyn(dyn, ustrip(uconvert(u"K", temperature)), settings)
end


