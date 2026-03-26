module Assignment_3_2

using BenchmarkTools

# Import local module
include("../helpers/__init__.jl")

# Import local module (assignment 3 helpers)
include("helpers/__init__.jl")


DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_3"


function main(;
    do_bench::Bool=false,
    do_cache::Bool=false,
    plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR,
)
    if do_bench
        @time "Done benchmarking" begin
            # TODO
        end
    end

    # TODO

    return
end

end # module
