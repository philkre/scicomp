using ArgParse
# using Debugger

include("src/helpers/__init__.jl")
using .Helpers.DistributedUtil: maximize_workers

# Globals
"Do benchmarking default behavior"
do_bench::Bool = false

"Do GIF creation for wave equation default behavior"
do_gif::Bool = false

"Use caching for diffusion simulation default behavior"
do_cache::Bool = false

"Output directory for plots"
plot_output_dir = "plots/ass_1/"

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
        "-p"
        arg_type = String
        help = "Output directory for plots"
        default = plot_output_dir
        # "arg1"
        # help = "a positional argument"
        # required = true
    end
    return parse_args(s)
end


if abspath(PROGRAM_FILE) == @__FILE__
    # Parse arguments using ArgParse
    @info args = parse_commandline()


    # Add workers for distributed computing
    @time "Added workers" maximize_workers()
end