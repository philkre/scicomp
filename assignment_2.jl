import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using ArgParse
using Distributed: @everywhere

include("src/helpers/__init__.jl")
using .Helpers.DistributedUtil: set_procs, maximize_workers

# Globals
"Do benchmarking (default behavior)"
do_bench::Bool = false

"Do GIF creation for wave equation (default behavior)"
do_gif::Bool = false

"Use caching for diffusion simulation (default behavior)"
do_cache::Bool = false

"Output directory for plots"
plot_output_dir = "plots/ass_2/"


# Do assignments (default behavior)
do_ass_1 = true
do_ass_2 = true
do_ass_3 = true


"""
    parse_commandline()::Dict{String,Any}

Parse command-line arguments for controlling assignment execution.

# Available flags:
- `--bench`, `-b`: Execute benchmarking
- `--gif`, `-g`: Create GIFs
- `--cache`, `-c`: Use caching
- `--backend`: Backend for Assignment 2.1 (`auto`, `cpu`, `metal`, `cuda`)
- `--solver`: Solver for Assignment 2.x (`sor`, `rb_sor`, `multigrid`)
- `-p`: Output directory for plots (default: "plots/ass_2/")
- `--ass1`: Execute only Assignment 2.1
- `--ass2`: Execute only Assignment 2.2
- `--ass3`: Execute only Assignment 2.3

# Returns
- `Dict{String,Any}`: Dictionary with parsed argument values
"""
function parse_commandline()::Dict{String,Any}
    s = ArgParseSettings()

    @add_arg_table s begin
        "--bench", "-b"
        action = :store_true
        help = "execute benchmarking"

        "--gif", "-g"
        help = "create GIFs"
        action = :store_true

        "--cache", "-c"
        help = "use caching"
        action = :store_true

        "--gpu"
        help = "use GPU for computation. WARNING: This feature currently only supports Metal.jl on macOS."
        action = :store_true

        "--backend"
        help = "backend for Assignment 2.1 (`auto`, `cpu`, `metal`, `cuda`). `auto` keeps legacy behavior via --gpu."
        arg_type = String
        default = "auto"

        "--solver"
        help = "solver for Assignment 2.x (`sor`, `rb_sor`, `multigrid`)."
        arg_type = String
        default = "rb_sor"

        "--nprocs"
        help = "number of processes for distributed computing"
        arg_type = Int

        "-p"
        arg_type = String
        help = "Output directory for plots"
        default = plot_output_dir

        "--ass1"
        help = "execute (only) Assignment 2.1"
        action = :store_true

        "--ass2"
        help = "execute (only) Assignment 2.2"
        action = :store_true

        "--ass3"
        help = "execute (only) Assignment 2.3"
        action = :store_true


        # "arg1"
        # help = "a positional argument"
        # required = true
    end
    return parse_args(s)
end


if ((abspath(PROGRAM_FILE) == @__FILE__) || !isempty(PROGRAM_FILE)) && !isinteractive()
    # Parse arguments using ArgParse
    args = parse_commandline()
    do_bench = args["bench"]
    do_gif = args["gif"]
    do_cache = args["cache"]
    use_GPU = args["gpu"]
    backend_arg = lowercase(args["backend"])
    if backend_arg ∉ ("auto", "cpu", "metal", "cuda")
        error("Invalid --backend value '$backend_arg'. Valid options are: auto, cpu, metal, cuda.")
    end
    backend_ass2_1::Symbol = Symbol(backend_arg == "auto" ? (use_GPU ? "metal" : "cpu") : backend_arg)
    use_GPU_runtime::Bool = backend_ass2_1 == :metal
    if backend_ass2_1 == :cuda
        @info "backend=:cuda selected. Assignment 2.x will use CUDA-backed solvers where available."
    end
    if backend_arg != "auto" && use_GPU
        @info "--backend overrides --gpu for Assignment 2.1 selection."
    end
    solver_arg = lowercase(args["solver"])
    if solver_arg ∉ ("sor", "rb_sor", "multigrid")
        error("Invalid --solver value '$solver_arg'. Valid options are: sor, rb_sor, multigrid.")
    end
    solver_ass2::Symbol = Symbol(solver_arg)
    nprocs = args["nprocs"]
    plot_output_dir = args["p"]

    # Only ignore default behavior if any of the assignment flags are set, otherwise run all assignments by default
    _ass1 = args["ass1"]
    _ass2 = args["ass2"]
    _ass3 = args["ass3"]
    if any([_ass1, _ass2, _ass3])
        do_ass_1 = _ass1
        do_ass_2 = _ass2
        do_ass_3 = _ass3
    end

    # Add workers for distributed computing
    @time "Added workers" begin
        if (nprocs === nothing)
            maximize_workers()
        else
            set_procs(nprocs)
        end
    end

    # Assignment 2.1
    if do_ass_1
        @info "Loading Assignment 2.1 on all workers"
        @time "Loaded Assignment 2.1" begin
            @everywhere include("src/ass_2/assignment_2_1.jl")
            @everywhere using .Assignment_2_1: main as main_2_1
        end
        @time "Assignment 2.1 completed" main_2_1(;
            do_bench=do_bench,
            do_gif=do_gif,
            do_cache=do_cache,
            use_GPU=use_GPU_runtime,
            backend=backend_ass2_1,
            solver=solver_ass2,
            plot_output_dir=plot_output_dir,
        )
    end

    # Assignment 2.2
    if do_ass_2
        @info "Loading Assignment 2.2 on all workers"
        @time "Loaded Assignment 2.2" begin
            @everywhere include("src/ass_2/assignment_2_2.jl")
            @everywhere using .Assignment_2_2: main as main_2_2
        end
        @time "Assignment 2.2 completed" main_2_2(; do_bench=do_bench, do_gif=do_gif, do_cache=do_cache, use_GPU=use_GPU_runtime, backend=backend_ass2_1, solver=solver_ass2, plot_output_dir=plot_output_dir)
    end

    # Assignment 2.3
    if do_ass_3
        @info "Loading Assignment 2.3 on all workers"
        @time "Loaded Assignment 2.3" begin
            @everywhere include("src/ass_2/assignment_2_3.jl")
            @everywhere using .Assignment_2_3: main as main_2_3
        end
        @time "Assignment 2.3 completed" main_2_3(; do_bench=do_bench, do_gif=do_gif, do_cache=do_cache, plot_output_dir=plot_output_dir)
    end
end
