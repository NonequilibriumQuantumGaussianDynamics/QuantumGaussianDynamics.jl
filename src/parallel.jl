

function parallel_force_distribute(nconf)
    rank = 0
    size = 1
    root = 0
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
    end
    
    av_point = Int32(nconf ÷ size)
    rest = mod(nconf,size) 

    start_per_proc = Vector{Int32}(undef,size)
    end_per_proc = Vector{Int32}(undef,size)

    start = 1
    for i in 1:size
        start_per_proc[i] = start
        if rest == 0
            end_per_proc[i] = start + av_point -1
            end_ = start + av_point - 1
            start = end_ + 1
        else
            end_per_proc[i] = start + av_point 
            end_ = start + av_point 
            start = end_ + 1
            rest -= 1
        end
    end

    if MPI.Initialized()
        MPI.Barrier(comm)
    end
    if rank == root
        println("start", start_per_proc)
        println("end", end_per_proc)
    end 
    return start_per_proc, end_per_proc
end


function write_file(file,line)
    rank = 0
    size = 1
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
    end

    if rank == 0
        write(file, line) 
        flush(file)
    end
end

    

function init_file(filename) #TODO: TYPE INSTABILITY HERE
    if MPI.Initialized()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        root = 0
        if rank == 0 
            file = open(filename, "w")
            close(file)
            file = open(filename, "a")
            return file
        else
            return ""
        end
    else
        file = open(filename, "w")
        close(file)
        file = open(filename, "a")
        return file
    end

end
    
