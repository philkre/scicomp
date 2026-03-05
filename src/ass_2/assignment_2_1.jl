module Assignment_2_1

using Metal: MtlMatrix
using Plots: heatmap
using BenchmarkTools

# Import local module
include("../helpers/__init__.jl")

# Benchmarking
using .Helpers.Benchmark: bench_funcs

# Diffusion helpers
using .Helpers.Diffusion: solve_until_tol, c_next_SOR_sink!, c_next_SOR_sink_red_black!, solve_until_tol_metal!, c_next_SOR_sink_metal!
using .Helpers.DLAUtil: run_diffusion_limited_aggregation

# Plotting
using .Helpers.SaveFig: savefig_auto_folder
using .Helpers: get_heatmap_kwargs
using .Helpers.DistributedGIF: gif_slow, distributed_gif


FloatMatrix = Union{Matrix{Float64},Matrix{Float32}}
DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_2"

DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_2"


function run_bench(; N::Int=100, L::Float64=1.0, omega::Float64=1.85, tol::Float64=10^-6, use_GPU::Bool=false, plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR)
    # Create initial conditions
    c_0 = zeros(N, N)
    c_0[:, end] .= 1
    sink_mask = zeros(Bool, N, N)
    sink_mask[N÷2, 1] = true
    heatmap_kwargs = get_heatmap_kwargs(N, L)
    savefig_auto_folder(heatmap(c_0'; heatmap_kwargs...), joinpath(plot_output_dir, "bench_c_0_heatmap.png"))

    # Define testable functions
    test_funcs = Vector{Function}([c_next_SOR_sink_red_black!, c_next_SOR_sink!])

    # Test single iterations
    bench_funcs(test_funcs, copy(c_0), omega, sink_mask)
    # Test convergence
    for func in test_funcs
        @info "Benchmarking $func convergence"
        # Verify convergence rate
        c_new, _deltas = solve_until_tol(func, c_0, tol, 10_000, omega, sink_mask, quiet=false)
        # Save test image of equilibrium state
        savefig_auto_folder(heatmap(c_new'; heatmap_kwargs...), joinpath(plot_output_dir, "bench_c_0_$(func)_equilibrium_heatmap.png"))
        # Benchmark full convergence
        display(@benchmark solve_until_tol($func, $c_0, $tol, 10_000, $omega, $sink_mask, quiet=true))
        print("\n\n")
    end

    if use_GPU
        c_0 = MtlMatrix(Matrix{Float32}(c_0))
        sink_mask = MtlMatrix(Bool.(sink_mask))
        bench_funcs(Vector{Function}([c_next_SOR_sink_metal!]), copy(c_0), Float32(omega), sink_mask)
        @info "Benchmarking $c_next_SOR_sink_metal! convergence on GPU"
        # Verify convergence rate
        c_new = solve_until_tol_metal!(c_next_SOR_sink_metal!, copy(c_0), tol, 10_000, Float32(omega), sink_mask, quiet=false)
        # Save test image of equilibrium state
        savefig_auto_folder(heatmap(Array(c_new)'; heatmap_kwargs...), joinpath(plot_output_dir, "bench_c_0_$(c_next_SOR_sink_metal!)_equilibrium_heatmap.png"))
        # Benchmark full convergence on GPU
        display(@benchmark solve_until_tol_metal!($c_next_SOR_sink_metal!, copy($c_0), $tol, 10_000, Float32($omega), $sink_mask, quiet=true))
        print("\n\n")
    end
end


function main(; use_GPU::Bool=false, do_bench::Bool=false, do_gif::Bool=false, do_cache::Bool=false, plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR)
    N::Int = 100
    L = 1.0
    omega_sor = 1.91
    tol = 10^(-3)
    eta = 1.5
    i_max = 10_000
    frames = 1000

    if do_bench
        run_bench(; N=N, L=L, omega=omega_sor, tol=tol, use_GPU=use_GPU, plot_output_dir=plot_output_dir)
    end

    @time "Finished diffusion limited aggregation" run_diffusion_limited_aggregation(N, L, eta, tol, frames; i_max_conv=i_max, omega_sor=omega_sor, use_GPU=use_GPU, do_gif=do_gif, plot_output_dir=plot_output_dir)

    # TODO: find optimal omega (?not possible due to different size of mask each frame)
    # TODO: experiment with eta

    return
end

end # module