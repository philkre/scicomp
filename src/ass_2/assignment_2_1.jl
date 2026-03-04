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
using .Helpers.Diffusion: solve_until_tol, c_next_SOR_sink!

# Plotting
using .Helpers.SaveFig: savefig_auto_folder
using .Helpers: get_heatmap_kwargs
using .Helpers.DistributedGIF: gif_slow, distributed_gif


"""Get equilibrium concentration for initial condition"""
function get_c_0_eq(N::Int)::Matrix{Float64}
    c_0 = zeros(N, N)
    c_y = range(0, stop=1, length=N)
    c_0 .= c_y'  # Broadcast row vector to all rows
    return c_0
end

function c_next_SOR_sink_metal!(c::MtlArray{Float32,2}, omega::Float64, sink_mask::MtlArray{Bool,2})::MtlArray{Float32,2}
    N = size(c, 1)
    Fo = Float32(0.25 * omega)
    omega_f = Float32(omega)

    do_sink = any(sink_mask)

    # 2D configuration
    threads_per_group = (32, 32)  # 1024 threads total per group
    groups = (cld(N, 32), cld(N - 2, 32))  # Cover all rows and N-2 columns

    # Red-black ordering for in-place SOR updates
    # Red pass: update cells where (i+j) is even
    @metal threads = threads_per_group groups = groups c_next_SOR_kernel!(c, sink_mask, Fo, omega_f, N, do_sink, Int32(0))

    # Black pass: update cells where (i+j) is odd (no sync - let GPU schedule)
    @metal threads = threads_per_group groups = groups c_next_SOR_kernel!(c, sink_mask, Fo, omega_f, N, do_sink, Int32(1))

    # Only sync at the end
    Metal.synchronize()

    if do_sink
        c[sink_mask] .= 0.0f0
    end

    return c
end


function c_next_SOR_kernel!(c::MtlDeviceMatrix{Float32,1}, sink_mask::MtlDeviceMatrix{Bool,1}, Fo::Float32, omega::Float32, N::Int, do_sink::Bool, color::Int32)
    i = thread_position_in_grid_2d().x
    j = thread_position_in_grid_2d().y + 1  # Shift j by 1 to account for skipping first and last column

    # Red-black ordering: only update if (i+j) mod 2 matches color
    if (i + j) % 2 != color
        return
    end

    # Check bounds
    if i > N || j > N - 1 || i < 1 || j < 2
        return
    end

    # Check mask and exit early if this cell is a sink
    if do_sink && sink_mask[i, j]
        return
    end

    i_right = (i == N) ? 1 : i + 1
    i_left = (i == 1) ? N : i - 1

    c[i, j] = Fo * (
        c[i_right, j] +
        c[i_left, j] +
        c[i, j+1] +
        c[i, j-1]
    ) + (1 - omega) * c[i, j]
    return
end


function diffusion_limited_aggregation_step(
    c::Union{Matrix{Float64},MtlArray{Float32,2}},
    c_sink::Union{Matrix{Bool},MtlArray{Bool,2}},
    c_source::Matrix{Bool},
    eta::Float64;
    tol::Float64=1e-6,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.85,
    use_GPU::Bool=false)::Tuple{Union{Matrix{Float64},Matrix{Float32}},Matrix{Bool}}

    cpu_c = Array(c)
    cpu_sink = Array(c_sink)
    if use_GPU
        c, _deltas = solve_until_tol(c_next_SOR_sink_metal!, c, tol, i_max_conv, omega_sor; sink_mask=c_sink, quiet=true)
        # Transfer from GPU once per frame
        copyto!(cpu_c, c)
        copyto!(cpu_sink, c_sink)
    else
        c, _deltas = solve_until_tol(c_next_SOR_sink!, c, tol, i_max_conv, omega_sor, c_sink; quiet=true)
        cpu_c = c
        cpu_sink = c_sink
    end

    neighbor_mask = falses(size(c_sink))
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
    if !c_source[chosen_cell]
        # Strike the lightning: set chosen cell as new source
        cpu_sink[chosen_cell] = true
        # Update GPU array only if using GPU
        if use_GPU
            copyto!(c_sink, cpu_sink)
        end
    end

    return cpu_c, cpu_sink
end


function run_diffusion_limited_aggregation(N::Int, L::Float64, eta::Float64, tol::Float64, frames::Int; i_max_conv::Int=10_000, omega_sor::Float64, use_GPU::Bool=false, do_gif::Bool=false, plot_output_dir::String="plots/ass_2")
    # Instantiate starting conditions

    # Source
    c_source = zeros(Bool, N, N)
    c_source[:, end] .= 1
    # Sink
    c_sink_template = zeros(Bool, N, N)
    # Single seed
    c_sink_template[50, 1] = true
    c_sink = use_GPU ? MtlArray(copy(c_sink_template)) : copy(c_sink_template)
    # Start with equilibrium solution of initial conditions
    c_0_eq = get_c_0_eq(N)
    c = use_GPU ? MtlArray(Matrix{Float32}(c_0_eq)) : Matrix{Float64}(c_0_eq)
    @info "Created initial conditions"

    plots = Vector{Plots.Plot{Plots.GRBackend}}(undef, frames)
    heatmap_kwargs = get_heatmap_kwargs(N, L)

    # Pre-allocate CPU buffers to avoid repeated allocations
    cpu_sink = Array(c_sink)
    cpu_c = Array(c)

    @showprogress "Solving frames" for i in 1:1:frames
        cpu_c, cpu_sink = diffusion_limited_aggregation_step(c, c_sink, c_source, eta; tol=tol, i_max_conv=i_max_conv, omega_sor=omega_sor, use_GPU=use_GPU)

        if do_gif
            c_plot = max.(min.(cpu_c + cpu_sink, 1.0), 0)  # Cap concentration inside [0.0, 1.0] for better visualization
            p = heatmap(c_plot'; heatmap_kwargs...)
            plots[i] = p
        end
    end

    @time "Saved final state" begin
        c_plot = max.(min.(cpu_c + cpu_sink, 1.0), 0)  # Cap concentration inside [0.0, 1.0] for better visualization
        p = heatmap(c_plot'; heatmap_kwargs...)
        savefig_auto_folder(p, joinpath(plot_output_dir, "diffusion_limited_aggregation_end_N=$N.png"))
    end

    if do_gif
        @time "Finished gif generation" begin
            distributed_gif(plots, joinpath(plot_output_dir, "diffusion_limited_aggregation_N=$N.gif"); fps=60, do_palette=false, width=900, hwaccel="videotoolbox")
        end
    end
end


function c_next_SOR_sink_red_black!(c::Matrix{Float64}, omega::Float64, sink_mask::Matrix{Bool})::Matrix{Float64}
    N = size(c, 1)
    Fo = 0.25 * omega

    do_sink = any(sink_mask)

    # Red pass: update cells where for even rows
    @inbounds for i in 2:2:N-1
        @inbounds @turbo for j in 2:2:N-1  # Start at 2 step by 2
            c[i, j] = Fo * (
                c[i+1, j] +
                c[i-1, j] +
                c[i, j+1] +
                c[i, j-1]
            ) + (1 - omega) * c[i, j]
        end
    end
    # Red pass: update cells where for uneven rows
    @inbounds for i in 3:2:N-1
        @inbounds @turbo for j in 3:2:N-1  # Start at 3 step by 2
            c[i, j] = Fo * (
                c[i+1, j] +
                c[i-1, j] +
                c[i, j+1] +
                c[i, j-1]
            ) + (1 - omega) * c[i, j]
        end
    end

    # Black pass: update cells where (i+j) is odd
    @inbounds for i in 2:N-1
        @inbounds for j in (3-(i%2)):2:N-1  # Start at 2 or 3 based on i (opposite of red), step by 2
            c[i, j] = Fo * (
                c[i+1, j] +
                c[i-1, j] +
                c[i, j+1] +
                c[i, j-1]
            ) + (1 - omega) * c[i, j]
        end
    end

    # Do boundary pass - Red cells
    @inbounds @turbo for j in 2:2:N-1  # Even j values (since i=1 is odd)
        c[1, j] = Fo * (
            c[2, j] +
            c[N, j] +
            c[1, j+1] +
            c[1, j-1]
        ) + (1 - omega) * c[1, j]
    end
    @inbounds @turbo for j in 3:2:N-1  # Odd j values (since i=N parity depends on N)
        c[N, j] = Fo * (
            c[1, j] +
            c[N-1, j] +
            c[N, j+1] +
            c[N, j-1]
        ) + (1 - omega) * c[N, j]
    end

    # Do boundary pass - Black cells
    @inbounds @turbo for j in 3:2:N-1  # Odd j values
        c[1, j] = Fo * (
            c[2, j] +
            c[N, j] +
            c[1, j+1] +
            c[1, j-1]
        ) + (1 - omega) * c[1, j]
    end
    @inbounds @turbo for j in 2:2:N-1  # Even j values
        c[N, j] = Fo * (
            c[1, j] +
            c[N-1, j] +
            c[N, j+1] +
            c[N, j-1]
        ) + (1 - omega) * c[N, j]
    end

    # Apply sink mask after both passes
    if do_sink
        c[sink_mask] .= 0.0
    end

    return c
end


function main(; use_GPU::Bool=false, do_bench::Bool=false, do_gif::Bool=false, do_cache::Bool=false, plot_output_dir::String="plots/ass_2")
    if do_bench
        N = 100
        L = 1.0
        omega = 1.85
        c_0 = zeros(N, N)
        c_0[:, end] .= 1
        sink_mask = zeros(Bool, N, N)
        sink_mask[50, 1] = true
        tol = 10^-6
        heatmap_kwargs = get_heatmap_kwargs(N, L)
        savefig(heatmap(c_0'; heatmap_kwargs...), joinpath(plot_output_dir, "c_0_heatmap.png"))

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
            savefig(heatmap(c_new'; heatmap_kwargs...), joinpath(plot_output_dir, "c_0_$(func)_equilibrium_heatmap.png"))
            # Benchmark full convergence
            display(@benchmark solve_until_tol($func, $c_0, $tol, 10_000, $omega, $sink_mask, quiet=true))
            print("\n\n")
        end

        if use_GPU
            c_0 = MtlArray(Matrix{Float32}(c_0))
            sink_mask = MtlArray(sink_mask)
            bench_funcs(Vector{Function}([c_next_SOR_sink_metal!]), copy(c_0), omega, sink_mask)
            @info "Benchmarking $c_next_SOR_sink_metal! convergence on GPU"
            # Verify convergence rate
            c_new, _deltas = solve_until_tol(c_next_SOR_sink_metal!, c_0, tol, 10_000, omega, sink_mask, quiet=false)
            # Save test image of equilibrium state
            savefig(heatmap(Array(c_new)'; heatmap_kwargs...), joinpath(plot_output_dir, "c_0_$(c_next_SOR_sink_metal!)__equilibrium_heatmap.png"))
            # Benchmark full convergence on GPU
            display(@benchmark solve_until_tol($c_next_SOR_sink_metal!, $c_0, $tol, 10_000, $omega, $sink_mask, quiet=true))
            print("\n\n")
        end
    end
    return
    N::Int = 100
    L = 1.0
    eta = 1.5
    tol = 10^(-3)
    i_max = 10_000
    omega_sor = 1.85
    frames = 1000

    run_diffusion_limited_aggregation(N, L, eta, tol, frames; i_max_conv=i_max, omega_sor=omega_sor, use_GPU=use_GPU, do_gif
        =do_gif, plot_output_dir=plot_output_dir)


    # TODO: find optimal omega (?not possible due to different size of mask each frame)
    # TODO: experiment with eta


end

end # module