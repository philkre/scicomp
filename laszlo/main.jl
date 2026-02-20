using Distributed


# Globals
"Do benchmarking"
do_bench::Bool = false
"Do assignment 1"
do_ass_1 = false
"Do assignment 2"
do_ass_2 = true


if abspath(PROGRAM_FILE) == @__FILE__
    do_bench = if length(ARGS) > 0 && "bench" in ARGS
        true
    else
        false
    end

    # Add worker processes first
    @info "Adding worker processes..."
    addprocs(Sys.CPU_THREADS - nprocs() - 1)
    @info "Number of workers: " * string(nprocs()) * "\nNumber of CPU threads: " * string(Sys.CPU_THREADS) * "\n"

    # Assignment 1.1
    if do_ass_1
        # Load the module on all workers
        @info "Loading Assignment 1.1 module on all workers..."
        @everywhere include("assignment_1_1.jl")
        @everywhere using .Assignment_1_1: main as main_1_1

        # Run the main function
        @info "Running main function for Assignment 1.1..."
        @time "Assigment 1.1 completed" main_1_1(do_bench=do_bench)
    end


    # Assignment 1.2
    if do_ass_2
        # Load the module on all workers
        @info "Loading Assignment 1.2 module on all workers..."
        @everywhere include("assignment_1_2.jl")
        @everywhere using .Assignment_1_2: main as main_1_2

        # Run the main function
        @info "Running main function for Assignment 1.2..."
        @time "Assigment 1.2 completed" main_1_2(do_bench=do_bench)
    end
end