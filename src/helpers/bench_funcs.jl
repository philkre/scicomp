module Benchmark

using BenchmarkTools

"""
    bench_funcs(funcs::Vector{Function}, args...; kwargs...)::Nothing

Benchmark multiple functions using `BenchmarkTools` and display the results.

Iterates through a vector of functions, benchmarking each one with the provided
arguments and keyword arguments, then displays the benchmark results.

# Arguments
- `funcs::Vector{Function}`: Vector of functions to benchmark
- `args...`: Positional arguments to pass to each function
- `kwargs...`: Keyword arguments to pass to each function

# Returns
- `Nothing`
"""
function bench_funcs(funcs::Vector{Function}, args...; kwargs...)::Nothing
    for func in (funcs)
        @info "Benchmarking $func..."
        display(@benchmark $func($args...; $kwargs...))
        print("\n\n")
    end
end

end # module