module DistributedUtil

using Distributed: addprocs, nprocs

"""
    maximize_workers(; quiet=false)

Add worker processes to utilize all available CPU threads for distributed computing.

Adds worker processes up to the number of CPU threads available, leaving one thread
for the main process. Optionally displays information about the number of workers
and CPU threads.

# Keyword Arguments
- `quiet::Bool`: If `false`, prints information messages about workers and CPU threads (default: `false`)

# Returns
- `Nothing`
"""
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