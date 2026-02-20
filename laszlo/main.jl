using Distributed

if abspath(PROGRAM_FILE) == @__FILE__
    # Add worker processes first
    addprocs(Sys.CPU_THREADS - nprocs() - 1)

    # Load the module on all workers
    @everywhere include("assignment_1.1.jl")
    @everywhere using .Assignment_1_1: main as main_1_1

    # Run the main function
    main_1_1()
end