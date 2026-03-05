using Distributed

# Globals
"Do benchmarking default behavior"
do_bench::Bool = false
"Do GIF creation for wave equation default behavior"
do_gif::Bool = false
"Use caching for diffusion simulation default behavior"
do_cache::Bool = false
"Do assignment 1"
do_ass_1 = true
"Do assignment 2"
do_ass_2 = true
"Do assignment 6"
do_ass_6 = true


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 0
        if "bench" in ARGS
            do_bench = true
        end

        if "gif" in ARGS
            do_gif = true
        end
    end

    # Add worker processes first
    @info "Adding worker processes..."
    addprocs(Sys.CPU_THREADS - nprocs() - 1)
    @info "Number of workers: " * string(nprocs()) * "\nNumber of CPU threads: " * string(Sys.CPU_THREADS) * "\n"

    # Assignment 1.1
    if do_ass_1
        # Load the module on all workers
        @info "Loading Assignment 1.1 module on all workers..."
        @time "Loaded Assignment 1.1 module" begin
            @everywhere include("assignment_1_1.jl")
            @everywhere using .Assignment_1_1: main as main_1_1
        end

        # Run the main function
        @info "Running main function for Assignment 1.1..."
        @time "Assigment 1.1 completed" main_1_1(; do_bench=do_bench, do_gif=do_gif)
    end


    # Assignment 1.2
    if do_ass_2
        # Load the module on all workers
        @info "Loading Assignment 1.2 module on all workers..."
        @time "Loaded Assignment 1.2 module" begin
            @everywhere include("assignment_1_2.jl")
            @everywhere using .Assignment_1_2: main as main_1_2
        end

        # Run the main function
        @info "Running main function for Assignment 1.2..."
        @time "Assigment 1.2 completed" main_1_2(; do_bench=do_bench, do_cache=do_cache, do_gif=do_gif)
    end


    # Assigment 1.6
    if do_ass_6
        # Load the module on all workers
        @info "Loading Assignment 1.6 module on all workers..."
        @time "Loaded Assignment 1.6 module" begin
            @everywhere include("assignment_1_6.jl")
            @everywhere using .Assignment_1_6: main as main_1_6
        end

        # Run the main function
        @info "Running main function for Assignment 1.6..."
        @time "Assigment 1.6 completed" main_1_6(; do_bench=do_bench)
    end
end