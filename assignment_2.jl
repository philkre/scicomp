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
- `--bench-dla-scaling`: Run scaling benchmark experiment for Assignment 2.1
- `--n-min`, `--n-max`, `--n-step`: N-range controls for scaling benchmark
- `--repeats`: Repeats per N/solver/backend for scaling benchmark
- `--frames-bench`: Frames used by scaling benchmark timing runs
- `--eta-dimension-sweep`: Run the eta sweep / fractal-dimension analysis for Assignment 2.1
- `--rerender-eta-dimension-plot`: Rebuild the eta sweep figure from existing CSV summaries
- `--eta-min`, `--eta-max`, `--eta-step`: Eta-range controls for the fractal-dimension sweep
- `--eta-repeats`: Repeats per eta value for the fractal-dimension sweep
- `--ps-dimension-sweep`: Run the stickiness sweep / fractal-dimension analysis for Assignment 2.2
- `--rerender-ps-dimension-plot`: Rebuild the p_s sweep figure from existing CSV summaries
- `--ps-min`, `--ps-max`, `--ps-step`: p_s-range controls for the Monte Carlo fractal-dimension sweep
- `--ps-repeats`: Repeats per p_s value for the Monte Carlo fractal-dimension sweep
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

        "--bench-dla-scaling"
        help = "run DLA scaling benchmark experiment (Assignment 2.1)"
        action = :store_true

        "--n-min"
        help = "minimum N for scaling benchmark"
        arg_type = Int
        default = 40

        "--n-max"
        help = "maximum N for scaling benchmark"
        arg_type = Int
        default = 200

        "--n-step"
        help = "N step for scaling benchmark"
        arg_type = Int
        default = 20

        "--repeats"
        help = "number of repeats per N/solver/backend for scaling benchmark"
        arg_type = Int
        default = 20

        "--frames-bench"
        help = "number of DLA frames per timing run in scaling benchmark"
        arg_type = Int
        default = 200

        "--eta-dimension-sweep"
        help = "run eta sweep / fractal-dimension analysis (Assignment 2.1)"
        action = :store_true

        "--rerender-eta-dimension-plot"
        help = "rebuild the eta sweep figure from existing CSV summaries (Assignment 2.1)"
        action = :store_true

        "--eta-min"
        help = "minimum eta for fractal-dimension sweep"
        arg_type = Float64
        default = 0.1

        "--eta-max"
        help = "maximum eta for fractal-dimension sweep"
        arg_type = Float64
        default = 2.0

        "--eta-step"
        help = "eta step for fractal-dimension sweep"
        arg_type = Float64
        default = 0.1

        "--eta-repeats"
        help = "number of repeats per eta value for the fractal-dimension sweep"
        arg_type = Int
        default = 30

        "--ps-dimension-sweep"
        help = "run p_s sweep / fractal-dimension analysis (Assignment 2.2)"
        action = :store_true

        "--rerender-ps-dimension-plot"
        help = "rebuild the p_s sweep figure from existing CSV summaries (Assignment 2.2)"
        action = :store_true

        "--ps-min"
        help = "minimum p_s for Monte Carlo fractal-dimension sweep"
        arg_type = Float64
        default = 0.1

        "--ps-max"
        help = "maximum p_s for Monte Carlo fractal-dimension sweep"
        arg_type = Float64
        default = 1.0

        "--ps-step"
        help = "p_s step for Monte Carlo fractal-dimension sweep"
        arg_type = Float64
        default = 0.1

        "--ps-repeats"
        help = "number of repeats per p_s value for the Monte Carlo fractal-dimension sweep"
        arg_type = Int
        default = 30

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
    do_bench_dla_scaling = args["bench-dla-scaling"]
    do_eta_dimension_sweep = args["eta-dimension-sweep"]
    rerender_eta_dimension_plot = args["rerender-eta-dimension-plot"]
    do_ps_dimension_sweep = args["ps-dimension-sweep"]
    rerender_ps_dimension_plot = args["rerender-ps-dimension-plot"]
    backend_arg = lowercase(args["backend"])
    if backend_arg ∉ ("auto", "cpu", "metal", "cuda")
        error("Invalid --backend value '$backend_arg'. Valid options are: auto, cpu, metal, cuda.")
    end
    backend_ass2_1::Symbol = Symbol(backend_arg)
    use_GPU_runtime::Bool = backend_ass2_1 == :metal || (backend_ass2_1 == :auto && use_GPU)
    if backend_ass2_1 == :cuda
        @info "backend=:cuda selected. Assignment 2.x will use CUDA-backed solvers where available."
    end
    if backend_ass2_1 == :auto && use_GPU
        @info "--gpu is ignored for backend auto-detection. Use --backend metal to force Metal."
    elseif backend_arg != "auto" && use_GPU
        @info "--backend overrides --gpu selection."
    end
    solver_arg = lowercase(args["solver"])
    if solver_arg ∉ ("sor", "rb_sor", "multigrid")
        error("Invalid --solver value '$solver_arg'. Valid options are: sor, rb_sor, multigrid.")
    end
    solver_ass2::Symbol = Symbol(solver_arg)
    n_min = args["n-min"]
    n_max = args["n-max"]
    n_step = args["n-step"]
    repeats = args["repeats"]
    frames_bench = args["frames-bench"]
    eta_min = args["eta-min"]
    eta_max = args["eta-max"]
    eta_step = args["eta-step"]
    eta_repeats = args["eta-repeats"]
    ps_min = args["ps-min"]
    ps_max = args["ps-max"]
    ps_step = args["ps-step"]
    ps_repeats = args["ps-repeats"]
    if n_step <= 0
        error("--n-step must be > 0")
    end
    if n_min > n_max
        error("--n-min must be <= --n-max")
    end
    if eta_step <= 0
        error("--eta-step must be > 0")
    end
    if eta_min > eta_max
        error("--eta-min must be <= --eta-max")
    end
    if ps_step <= 0
        error("--ps-step must be > 0")
    end
    if ps_min > ps_max
        error("--ps-min must be <= --ps-max")
    end
    if ps_min <= 0.0 || ps_max > 1.0
        error("--ps-min and --ps-max must satisfy 0 < p_s <= 1")
    end
    n_values = n_min:n_step:n_max
    eta_values = eta_min:eta_step:eta_max
    p_s_values = ps_min:ps_step:ps_max
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
            do_bench_dla_scaling=do_bench_dla_scaling,
            do_eta_dimension_sweep=do_eta_dimension_sweep,
            rerender_eta_dimension_plot=rerender_eta_dimension_plot,
            n_values=n_values,
            repeats=repeats,
            frames_bench=frames_bench,
            eta_values=eta_values,
            eta_repeats=eta_repeats,
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
        @time "Assignment 2.2 completed" main_2_2(;
            do_bench=do_bench,
            do_gif=do_gif,
            do_cache=do_cache,
            use_GPU=use_GPU_runtime,
            backend=backend_ass2_1,
            solver=solver_ass2,
            do_ps_dimension_sweep=do_ps_dimension_sweep,
            rerender_ps_dimension_plot=rerender_ps_dimension_plot,
            p_s_values=p_s_values,
            p_s_repeats=ps_repeats,
            plot_output_dir=plot_output_dir,
        )
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
