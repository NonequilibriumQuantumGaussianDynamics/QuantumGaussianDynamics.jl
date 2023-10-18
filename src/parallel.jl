
"""
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
root = 0
println("hereeeee")
println("rabk ", rank)
if size>1
    isMPI = true
else
    isMPI = false
end
export comm
export rank
export size
export root
export isMPI
"""

function parallel_force_distribute(nconf) where {T <: AbstractFloat}
    
    av_point = Int32(floor(nconf/size))
    rest = mod(nconf,size) 

    start_per_proc = Vector{Int32}(undef,size)
    end_per_proc = Vector{Int32}(undef,size)

    start = 1
    for i in 1:size
        start_per_proc[i] = start
        if rest == 0
            end_per_proc[i] = start + av_point 
            end_ = start + av_point 
            start = end_ + 1
        else
            end_per_proc[i] = start + av_point + 1
            end_ = start + av_point + 1
            start = end_ + 1
            rest -= 1
        end
    end
    println("here ", rank, root)
    error()
    if rank == root
        println("start", start_per_proc)
        println("end", end_per_proc)
    error()
    end 
    return start_per_proc, end_per_proc
end
    
