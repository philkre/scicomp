module Assignment_1_2

# Preperare distributed
using Distributed

# Benchmarking
include("helpers/benchmark.jl")

# Wave equation helpers
include("helpers/diffusion.jl")

# Plotting
using Plots
include("helpers/distributed_gif.jl")

function main(; do_bench::Bool=false)
    if do_bench
        # Setup for benchmarking
        N = 100
        L = 1.0
        dx = L / N
        D = 1.0
        dt = 0.00001
        c_0 = zeros(N, N)
        c_0[:, end] .= 1.0

        bench_funcs([c_next, c_next_single_loop, c_next_dist_turbo, c_next_dist], c_0, D, dx, dt)
    end

end

end