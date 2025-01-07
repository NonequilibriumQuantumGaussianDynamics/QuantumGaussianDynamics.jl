    

function Dynamics(dt:: Quantity, 
        total_time:: Quantity, 
        N :: Int;
        kwargs...)

    return Dynamics(ustrip(uconvert(u"fs", dt)), 
                    ustrip(uconvert(u"fs", total_time)), 
                    N;
                    kwargs...)
end


function init_from_dyn(dyn :: PyObject, temperature :: Quantity, settings :: Dynamics)
    return init_from_dyn(dyn, ustrip(uconvert(u"K", temperature)), settings)
end


