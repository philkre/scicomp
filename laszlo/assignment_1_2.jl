module Assignment_1_2

# Benchmarking
include("helpers/benchmark.jl")

# Diffusion helpers
include("helpers/diffusion.jl")

# Utilities
using JLD2
using Printf
using LaTeXStrings

# Plotting
using Plots
include("helpers/distributed_gif.jl")
include("helpers/heatmap_kwargs.jl")

function _resolve_output_path(output::String)::String
    if isabspath(output)
        return output
    end
    return normpath(joinpath(@__DIR__, output))
end

function _savefig(output::String)
    output_path = _resolve_output_path(output)
    mkpath(dirname(output_path))
    if isfile(output_path)
        rm(output_path; force=true)
    end
    savefig(output_path)
    isfile(output_path) || error("Failed to save plot to $output_path")
    return output_path
end

function _savefig(p, output::String)
    output_path = _resolve_output_path(output)
    mkpath(dirname(output_path))
    if isfile(output_path)
        rm(output_path; force=true)
    end
    savefig(p, output_path)
    isfile(output_path) || error("Failed to save plot to $output_path")
    return output_path
end

function get_c_intervals(c0::Matrix{Float64}, intervals::Vector{Float64}; L::Float64, N::Int64, D::Float64, dt::Float64, t_0::Float64=0.0)::Vector{Matrix{Float64}}
    c = c0

    t_starts = [t_0; intervals[1:end-1]]
    results = []
    for (t_start, t_end) in zip(t_starts, intervals)
        c = propagate_c_diffusion(c, L, N, D, t_start, t_end, dt)
        push!(results, c)
    end

    return results
end


function plot_intervals(c0::Matrix, intervals::Vector{Float64}, results::Vector{Matrix{Float64}}; L::Float64, N::Int64, t_0::Float64=0.0)
    # Plot concentration along y-axis at x=0 for all time intervals
    y = range(0, stop=L, length=size(c0, 2))
    p = plot(xlims=(0, L), dpi=300)
    for (t_end, c) in zip(intervals, results)
        plot!(p, y, c[1, :], label="t = $(t_end) (numerical)", dpi=300)
    end
    xlabel!(p, L"y")
    ylabel!(p, L"c(y)")
    _savefig(p, "plots/diffusion_yprofile.png")
    return p
end


function add_analytical_plot!(plt::Plots.Plot{Plots.GRBackend}, intervals::Vector{Float64}, analytical_sol::Function; D::Float64, L::Float64, N::Int64)
    y = range(0, stop=L, length=N)

    # Match each analytical curve to the corresponding numerical curve colour.
    # Series order in `plt`: initial t0 first, then numerical interval curves.
    for (i, t) in enumerate(intervals)

        colour = if i <= length(plt.series_list)
            get(plt.series_list[i].plotattributes, :seriescolor, nothing)
        else
            nothing
        end

        if isnothing(colour)
            plot!(plt, y, analytical_sol(t, D, L, N), label="t = $(t) (analytical)", linestyle=:dash, dpi=300)
        else
            plot!(plt, y, analytical_sol(t, D, L, N), label="t = $(t) (analytical)", linestyle=:dash, dpi=300, seriescolor=colour)
        end
    end

    _savefig(plt, "plots/diffusion_yprofile_analytical.png")
    return plt
end

function plot_heatmaps(intervals::Vector{Float64}, results::Vector{Matrix{Float64}}; L::Float64, N::Int64)
    plots = []
    heatmap_kwargs = get_heatmap_kwargs(N, L)
    for (t_end, c) in zip(intervals, results)
        push!(plots, heatmap(c'; title="t = $(t_end)", heatmap_kwargs...))
    end
    plot(plots..., layout=(2, 2), size=(800, 800), dpi=300, suptitle="Diffusion process at different time points")
    _savefig("plots/diffusion_heatmaps.png")
end


function animate_diffusion(c_0::Matrix{Float64}; L::Float64, N::Int64, D::Float64, dt::Float64)
    heatmap_kwargs = get_heatmap_kwargs(N, L)

    plots = (() -> begin
        frames::Vector{Plots.Plot{Plots.GRBackend}} = []
        c_i = c_0

        # Use a non-linear time stepping to capture the early evolution better, while still reaching t=1.0 in a reasonable number of steps
        for i in 0:317
            t_prev = (i - 1)^2 * 0.00001
            t = (i)^2 * 0.00001
            if t >= 1.0
                break
            end

            c_i = propagate_c_diffusion(c_i, L, N, D, t_prev, t, dt)
            push!(frames, plot(heatmap(c_i', title=@sprintf("t = %0.3f", t); heatmap_kwargs...)))
        end
        frames
    end)()

    gif_slow(plots, _resolve_output_path("plots/diffusion_evolution.gif"), fps=30)
end

function plot_analytical_difference(c0::Matrix{Float64}, intervals::Vector{Float64}, results::Vector{Matrix{Float64}}, analytical_sol::Function; L::Float64, N::Int64, D::Float64)
    y = range(0, stop=L, length=size(c0, 2))
    plot(legend=true, xlims=(0, L), dpi=300)
    for (t_end, c) in zip(intervals, results)
        analytical_profile = analytical_sol(t_end, D, L, N)
        difference = c[1, :] - analytical_profile
        plot!(y, difference, label="t = $(t_end)")
    end

    xlabel!(L"y")
    ylabel!(L"\delta c(y)")
    _savefig("plots/diffusion_difference.png")
    return current()
end

function main(; do_bench::Bool=false, do_cache::Bool=false)
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

    N = 100
    L = 1.0
    dx = L / N
    D = 1.0
    t_0 = 0.0
    dt = 0.00001
    c_0 = zeros(N, N)
    c_0[:, end] .= 1.0
    t_intervals = [0.001, 0.01, 0.1, 1.0]

    if do_cache && isfile("diffusion_results.jld2")
        @info "Loading cached results from diffusion_results.jld2..."
        @load "diffusion_results.jld2" results
    else
        @info "Running diffusion simulation and plotting y-profile results..."
        @time "Ran diffusion simulations" results = get_c_intervals(c_0, t_intervals; L=L, N=N, D=D, dt=dt, t_0=t_0)

        if do_cache
            @save "diffusion_results.jld2" results
        end
    end

    # Plot y-profile
    @info "Plotting y-profile results..."
    plt_intervals = plot_intervals(c_0, t_intervals, results; L=L, N=N, t_0=t_0)
    add_analytical_plot!(plt_intervals, t_intervals, analytical_sol; D=D, L=L, N=N)

    plot_analytical_difference(c_0, t_intervals, results, analytical_sol; L=L, N=N, D=D)

    # Plot heatmaps for each interval
    @info "Plotting heatmaps for each interval..."
    plot_heatmaps(t_intervals, results; L=L, N=N)

    # Animate diffusion process
    @info "Animating diffusion process..."
    @time "Saved animated diffusion process" animate_diffusion(c_0; L=L, N=N, D=D, dt=dt)
end

end
