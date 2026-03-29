import Pkg
Pkg.activate(@__DIR__)


using ArgParse
using PythonCall
using Distributed: @everywhere

using Helpers.DistributedUtil: set_procs, maximize_workers

# Globals
"Do benchmarking (default behavior)"
do_bench::Bool = false

"Do GIF creation for wave equation (default behavior)"
do_gif::Bool = false

"Use caching for diffusion simulation (default behavior)"
do_cache::Bool = false

"Output directory for plots"
plot_output_dir = "plots/ass_3/"


# Do assignments (default behavior)
do_ass_1 = true
do_ass_2 = true


"""
    parse_commandline()::Dict{String,Any}

Parse command-line arguments for controlling assignment execution.

# Available flags:
- `--bench`, `-b`: Execute benchmarking
- `--gif`, `-g`: Create GIFs
- `--cache`, `-c`: Use caching
- `-p`: Output directory for plots (default: "plots/ass_3/")
- `--ass1`: Execute only Assignment 3.1
- `--ass2`: Execute only Assignment 3.2

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

        "--nprocs"
        help = "number of processes for distributed computing"
        arg_type = Int

        "-p"
        arg_type = String
        help = "Output directory for plots"
        default = plot_output_dir

        "--ass1"
        help = "execute (only) Assignment 3.1"
        action = :store_true

        "--ass2"
        help = "execute (only) Assignment 3.2"
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
    nprocs = args["nprocs"]
    plot_output_dir = args["p"]

    # Only ignore default behavior if any of the assignment flags are set, otherwise run all assignments by default
    _ass1 = args["ass1"]
    _ass2 = args["ass2"]
    if any([_ass1, _ass2])
        do_ass_1 = _ass1
        do_ass_2 = _ass2
    end

    # Add workers for distributed computing
    if (nprocs === nothing)
        @time "Added workers" maximize_workers()
    elseif (nprocs > 1)
        @time "Added workers" set_procs(nprocs)
    end

    # Assignment 3.1
    if do_ass_1
        @info "Loading Assignment 3.1 on all workers"
        @time "Loaded Assignment 3.1" begin
            @everywhere include("src/ass_3/assignment_3_1.jl")
            @everywhere using .Assignment_3_1: main as main_3_1
        end
        @time "Assignment 3.1 completed" main_3_1(;
            do_bench=do_bench,
            do_cache=do_cache,
            plot_output_dir=plot_output_dir,
        )
    end

    # Assignment 3.2
    if do_ass_2
        @info "Loading Assignment 3.2 on all workers"
        @time "Loaded Assignment 3.2" begin
            @everywhere include("src/ass_3/assignment_3_2.jl")
            @everywhere using .Assignment_3_2: main as main_3_2
        end
        @time "Assignment 3.2 completed" main_3_2(;
            do_bench=do_bench,
            do_cache=do_cache,
            plot_output_dir=plot_output_dir,
        )
    end
end
