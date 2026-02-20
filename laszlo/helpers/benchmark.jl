using BenchmarkTools

function bench_funcs(funcs::Vector{Function}, args...; kwargs...)::Nothing
    for func in (funcs)
        @info "Benchmarking $func..."
        display(@benchmark $func($args...; $kwargs...))
        print("\n\n")
    end
end