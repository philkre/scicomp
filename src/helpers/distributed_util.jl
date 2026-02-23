module DistributedUtil

using Distributed: addprocs, nprocs

function maximize_workers(; quiet=false)
    # Add worker processes first
    if !quiet
        @info "Adding worker processes..."
    end
    addprocs(Sys.CPU_THREADS - nprocs() - 1)
    if !quiet
        @info "Number of workers: " * string(nprocs()) * "\nNumber of CPU threads: " * string(Sys.CPU_THREADS) * "\n"
    end
end

end # module