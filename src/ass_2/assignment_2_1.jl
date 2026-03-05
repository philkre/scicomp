module Assignment_2_1

using Metal
using Distributions
using Plots
using ProgressMeter
using LoopVectorization
using BenchmarkTools

# Import local module
include("../helpers/__init__.jl")

# Benchmarking
using .Helpers.Benchmark: bench_funcs


# Diffusion helpers
using .Helpers.Diffusion: solve_until_tol, c_next_SOR_sink!, c_next_SOR_sink_red_black!, solve_until_tol_metal!, c_next_SOR_sink_metal!

# Plotting
using .Helpers.SaveFig: savefig_auto_folder
using .Helpers: get_heatmap_kwargs
using .Helpers.DistributedGIF: gif_slow, distributed_gif


FloatMatrix = Union{Matrix{Float64},Matrix{Float32}}
DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_2"

"""Get equilibrium concentration for initial condition"""
function get_c_0_eq(N::Int)::Matrix{Float64}
    c_0 = zeros(N, N)
    c_y = range(0, stop=1, length=N)
    c_0 .= c_y'  # Broadcast row vector to all rows
    return c_0
end


function superimpose_c_sink(c::FloatMatrix, c_sink::Matrix{Bool})::FloatMatrix
    c_plot = copy(c)
    c_plot[c_sink] .= 1.0  # Cap concentration inside [0.0, 1.0] for better visualization
    return max.(min.(c_plot, 1.0), 0.0)
end


function diffusion_limited_aggregation_step!(
    c::Union{Matrix{Float64},MtlMatrix{Float32,Metal.PrivateStorage}},
    c_sink::Union{Matrix{Bool},MtlMatrix{Bool,Metal.PrivateStorage}},
    c_source::Matrix{Bool},
    eta::Float64,
    cpu_c::FloatMatrix,
    cpu_sink::Matrix{Bool}
    ;
    tol::Float64=1e-6,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.9,
    use_GPU::Bool=false,
    c_old::Union{MtlMatrix{Float32,Metal.PrivateStorage},Nothing}=nothing,
    diffs::Union{MtlMatrix{Float32,Metal.PrivateStorage},Nothing}=nothing)
    if use_GPU
        copyto!(cpu_c, solve_until_tol_metal!(c_next_SOR_sink_metal!, c, tol, i_max_conv, Float32(omega_sor), c_sink; check_every=25, c_old=c_old, diffs=diffs, quiet=true))
    else
        c = solve_until_tol(c_next_SOR_sink_red_black!, c, tol, i_max_conv, omega_sor, c_sink; quiet=true)[1]
        copyto!(cpu_c, c)
    end

    neighbor_mask = falses(size(cpu_sink))
    # Copy source stamp to neighbor candidates
    neighbor_mask[2:end, :] .|= cpu_sink[1:end-1, :]    # below source
    neighbor_mask[1:end-1, :] .|= cpu_sink[2:end, :]    # above source
    neighbor_mask[:, 2:end] .|= cpu_sink[:, 1:end-1]    # right of source
    neighbor_mask[:, 1:end-1] .|= cpu_sink[:, 2:end]    # left of source
    # Exclude original source cells from neighbors
    neighbor_mask .&= .!cpu_sink

    candidates = findall(neighbor_mask)
    # diffusion probabilities are proportional to concentration at candidate cells
    # Raise to the power eta, control floating point issues by maxing with 0.0
    probabilities = max.(cpu_c[candidates], 0.0) .^ eta
    # Normalize
    probabilities ./= sum(probabilities)
    # Choose candidate cell based on probabilities
    chosen_index = rand(Categorical(probabilities))
    chosen_cell = candidates[chosen_index]

    # Not allowed to change source into sink
    if c_source[chosen_cell]
        return
    end

    # Strike the lightning: set chosen cell as new source
    cpu_sink[chosen_cell] = true
    copyto!(c_sink, cpu_sink)

    return
end


function run_diffusion_limited_aggregation(N::Int, L::Float64, eta::Float64, tol::Float64, frames::Int; i_max_conv::Int=10_000, omega_sor::Float64, use_GPU::Bool=false, do_gif::Bool=false, plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR)
    # Instantiate starting conditions

    # Source
    c_source = zeros(Bool, N, N)
    c_source[:, end] .= 1
    # Sink
    c_sink = zeros(Bool, N, N)
    # Single seed
    c_sink[N÷2, 1] = true
    # Start with equilibrium solution of initial conditions
    c = get_c_0_eq(N)

    c_old::Union{MtlMatrix{Float32,Metal.PrivateStorage},Nothing} = nothing
    diffs::Union{MtlMatrix{Float32,Metal.PrivateStorage},Nothing} = nothing

    if use_GPU
        c_sink = MtlMatrix(c_sink)
        c = MtlMatrix(Matrix{Float32}(c))
        # Pre allocate GPU matrices
        c_old = similar(c)
        diffs = similar(c)
    end

    # Allocate once
    cpu_c = Array(c)
    cpu_sink = Array(c_sink)

    @info "Created initial conditions"

    # Allocate plots vector
    plots = Vector{Plots.Plot{Plots.GRBackend}}(undef, frames)
    # Fetch plotting kwargs
    heatmap_kwargs = get_heatmap_kwargs(N, L)

    @showprogress "Solving frames" for i in 1:1:frames
        diffusion_limited_aggregation_step!(c, c_sink, c_source, eta, cpu_c, cpu_sink; tol=tol, i_max_conv=i_max_conv, omega_sor=omega_sor, use_GPU=use_GPU, c_old=c_old, diffs=diffs)

        if do_gif
            c_plot = superimpose_c_sink(cpu_c, cpu_sink)
            p = heatmap(c_plot'; heatmap_kwargs...)
            plots[i] = p
        end
    end

    # Save final state plot
    @time "Saved final state" begin
        c_plot = superimpose_c_sink(cpu_c, cpu_sink)
        p = heatmap(c_plot'; heatmap_kwargs...)
        savefig_auto_folder(p, joinpath(plot_output_dir, "diffusion_limited_aggregation_end_N=$N.png"))
    end

    # Save gif of the process
    if do_gif
        @time "Finished gif generation" begin
            distributed_gif(plots, joinpath(plot_output_dir, "diffusion_limited_aggregation_N=$N.gif"); fps=60, do_palette=true, width=900, hwaccel="videotoolbox")
        end
    end

    return
end


function run_bench(; N::Int=100, L::Float64=1.0, omega::Float64=1.85, tol::Float64=10^-6, use_GPU::Bool=false, plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR)
    # Create initial conditions
    c_0 = zeros(N, N)
    c_0[:, end] .= 1
    sink_mask = zeros(Bool, N, N)
    sink_mask[N÷2, 1] = true
    heatmap_kwargs = get_heatmap_kwargs(N, L)
    savefig(heatmap(c_0'; heatmap_kwargs...), joinpath(plot_output_dir, "bench_c_0_heatmap.png"))

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
        savefig(heatmap(c_new'; heatmap_kwargs...), joinpath(plot_output_dir, "bench_c_0_$(func)_equilibrium_heatmap.png"))
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
        savefig(heatmap(Array(c_new)'; heatmap_kwargs...), joinpath(plot_output_dir, "bench_c_0_$(c_next_SOR_sink_metal!)_equilibrium_heatmap.png"))
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