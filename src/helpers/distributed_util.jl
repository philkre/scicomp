module DistributedUtil

using Distributed: addprocs, nprocs

function set_procs(n_procs_max::Int)
    current_procs = nprocs()
    if current_procs < n_procs_max
        addprocs(n_procs_max - current_procs - 1)
    else
        @info "Already have $current_procs workers, which is >= requested $n_procs_max. No new workers added."
    end
end

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
    set_procs(Sys.CPU_THREADS)
    if !quiet
        @info "Number of workers: " * string(nprocs()) * "\nNumber of CPU threads: " * string(Sys.CPU_THREADS) * "\n"
    end
end

end # module