module DLAUtil

import Plots
import Metal
import PythonPlot
using Metal: MtlMatrix, PrivateStorage
import CUDA
using CUDA: CuArray
using ProgressMeter: @showprogress
using Printf: @sprintf
using Random: seed!
using Statistics: mean, std
using Plots: heatmap, plot, plot!, scatter!, Plot, GRBackend, palette
using LaTeXStrings: @L_str

include("diffusion.jl")
using .Diffusion: c_anal_2d

include("DLA_core.jl")
using .DLACore: FloatMatrix, diffusion_limited_aggregation_step!, choose_candidate, choose_candidate_monte_carlo

include("savefig.jl")
using .SaveFig: savefig_auto_folder

include("distributed_gif.jl")
using .DistributedGIF: distributed_gif

include("get_heatmap_kwargs.jl")


function _detect_best_backend()::Symbol
    if Sys.isapple() && isdefined(Metal, :functional) && Metal.functional()
        return :metal
    end
    if isdefined(CUDA, :functional) && CUDA.functional()
        return :cuda
    end
    return :cpu
end


function _validate_backend(backend::Symbol)::Symbol
    if backend == :auto
        return _detect_best_backend()
    end
    if backend == :cpu
        return :cpu
    end
    if backend == :metal
        Metal.functional() || error("backend=:metal requested but Metal is not functional.")
        return :metal
    end
    if backend == :cuda
        CUDA.functional() || error("backend=:cuda requested but CUDA is not functional.")
        return :cuda
    end
    throw(ArgumentError("Invalid backend=$backend. Valid options: :auto, :cpu, :metal, :cuda"))
end


function _valid_solver_backend_pairs(; backend::Symbol=:auto)::Vector{NamedTuple{(:backend, :solver),Tuple{Symbol,Symbol}}}
    pairs = NamedTuple{(:backend, :solver),Tuple{Symbol,Symbol}}[]
    append!(pairs, [(backend=:cpu, solver=:sor), (backend=:cpu, solver=:rb_sor), (backend=:cpu, solver=:multigrid)])

    gpu_backend = backend == :auto ? _detect_best_backend() : _validate_backend(backend)
    if gpu_backend in (:metal, :cuda)
        append!(pairs, [(backend=gpu_backend, solver=:rb_sor), (backend=gpu_backend, solver=:multigrid)])
    end
    return pairs
end


function superimpose_c_sink(c::FloatMatrix, c_sink::Matrix{Bool})::FloatMatrix
    c_plot = copy(c)
    c_plot[c_sink] .= 1.0  # Cap concentration inside [0.0, 1.0] for better visualization
    return clamp!(c_plot, 0.0, 1.0)
end


function plot_DLA_frame(cpu_c::FloatMatrix, cpu_sink::Matrix{Bool}; heatmap_kwargs...)
    c_plot = superimpose_c_sink(cpu_c, cpu_sink)
    return heatmap(c_plot'; heatmap_kwargs...)
end


"""
    cluster_stats(mask; occupied_value=true, normalized=false)

Compute the particle count and gyration radius for an occupancy mask.

`occupied_value` specifies which entries correspond to occupied sites. If
`normalized=true`, coordinates are mapped to `[0, 1] x [0, 1]`; otherwise grid
indices are used directly.
"""
function cluster_stats(
    mask::AbstractMatrix;
    occupied_value=true,
    normalized::Bool=false,
)
    n_particles = count(==(occupied_value), mask)
    if n_particles == 0
        return 0, 0.0
    end

    ni, nj = size(mask)
    xs = Vector{Float64}(undef, n_particles)
    ys = Vector{Float64}(undef, n_particles)

    k = 1
    @inbounds for I in CartesianIndices(mask)
        if mask[I] == occupied_value
            i, j = Tuple(I)
            if normalized
                xs[k] = (i - 1) / max(1, ni - 1)
                ys[k] = (j - 1) / max(1, nj - 1)
            else
                xs[k] = i
                ys[k] = j
            end
            k += 1
        end
    end

    x_cm = sum(xs) / n_particles
    y_cm = sum(ys) / n_particles
    r2 = 0.0
    @inbounds @simd for idx in eachindex(xs)
        dx = xs[idx] - x_cm
        dy = ys[idx] - y_cm
        r2 += dx * dx + dy * dy
    end

    return n_particles, sqrt(r2 / n_particles)
end


function _linear_fit_slope(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})::Float64
    n = length(xs)
    n == length(ys) || throw(ArgumentError("xs and ys must have the same length."))
    n >= 2 || return NaN

    x_mean = mean(xs)
    y_mean = mean(ys)
    num = 0.0
    den = 0.0
    @inbounds @simd for i in eachindex(xs, ys)
        dx = xs[i] - x_mean
        num += dx * (ys[i] - y_mean)
        den += dx * dx
    end
    return den == 0.0 ? NaN : num / den
end


function _default_eta_backend(backend::Symbol)::Symbol
    return backend == :auto ? :cpu : _validate_backend(backend)
end


function _write_eta_csv(path::String, rows::Vector{NamedTuple}, header::String, row_formatter::Function)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, header)
        for row in rows
            println(io, row_formatter(row))
        end
    end
    return path
end


function _parameter_axis_label(parameter_name::Symbol)
    if parameter_name == :eta
        return L"\eta"
    elseif parameter_name == :p_s
        return L"p_s"
    end
    return string(parameter_name)
end


function _parameter_title_text(parameter_name::Symbol)
    if parameter_name == :eta
        return "eta"
    elseif parameter_name == :p_s
        return "p_s"
    end
    return string(parameter_name)
end


function _write_dimension_sweep_outputs(
    output_dir::String,
    prefix::String,
    parameter_name::Symbol,
    raw_rows::Vector{NamedTuple},
    trajectory_rows::Vector{NamedTuple},
    dimension_rows::Vector{NamedTuple},
)
    parameter_header = String(parameter_name)
    _write_eta_csv(
        joinpath(output_dir, prefix * "_raw.csv"),
        raw_rows,
        "$(parameter_header),repeat,step,particle_count,gyration_radius",
        row -> "$(getfield(row, parameter_name)),$(row.repeat),$(row.step),$(row.particle_count),$(row.gyration_radius)",
    )
    _write_eta_csv(
        joinpath(output_dir, prefix * "_trajectory_summary.csv"),
        trajectory_rows,
        "$(parameter_header),step,mean_particle_count,std_particle_count,mean_gyration_radius,std_gyration_radius,n_repeats",
        row -> "$(getfield(row, parameter_name)),$(row.step),$(row.mean_particle_count),$(row.std_particle_count),$(row.mean_gyration_radius),$(row.std_gyration_radius),$(row.n_repeats)",
    )
    _write_eta_csv(
        joinpath(output_dir, prefix * "_dimension_summary.csv"),
        dimension_rows,
        "$(parameter_header),mean_dimension,std_dimension,n_valid_runs",
        row -> "$(getfield(row, parameter_name)),$(row.mean_dimension),$(row.std_dimension),$(row.n_valid_runs)",
    )
    return nothing
end


function _read_eta_summary_csv(path::String, field_types::Vector{Pair{Symbol,DataType}})
    rows = NamedTuple[]
    open(path, "r") do io
        for (line_idx, line) in enumerate(eachline(io))
            line_idx == 1 && continue
            isempty(line) && continue
            values = split(line, ",")
            length(values) == length(field_types) || error("Malformed CSV row in $path: $line")

            names = Symbol[]
            parsed = Any[]
            for (idx, (name, T)) in enumerate(field_types)
                push!(names, name)
                push!(parsed, parse(T, values[idx]))
            end
            push!(rows, NamedTuple{Tuple(names)}(Tuple(parsed)))
        end
    end
    return rows
end


function _plot_dimension_sweep_experiment(
    trajectory_rows::Vector{NamedTuple},
    dimension_rows::Vector{NamedTuple};
    output_path::String,
    parameter_name::Symbol,
)
    parameter_values = sort(unique(getfield(row, parameter_name) for row in trajectory_rows))
    colors = palette(:viridis, max(length(parameter_values), 1))
    parameter_min = minimum(parameter_values)
    parameter_max = maximum(parameter_values)
    parameter_label = _parameter_axis_label(parameter_name)
    parameter_title = _parameter_title_text(parameter_name)

    p1 = plot(
        xlabel=L"R_g",
        ylabel=L"M",
        title="Particle Count vs Gyration Radius",
        legend=false,
        colorbar=false,
        dpi=300,
        left_margin=18Plots.mm,
        right_margin=10Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
    )

    p2 = plot(
        xlabel=parameter_label,
        ylabel=L"D",
        title="Fractal Dimension vs " * parameter_title,
        legend=false,
        dpi=300,
        left_margin=12Plots.mm,
        right_margin=14Plots.mm,
        bottom_margin=12Plots.mm,
        top_margin=6Plots.mm,
    )

    # Add a dummy series to drive a clean eta colorbar without creating a legend.
    scatter!(
        p1,
        [NaN],
        [NaN];
        marker_z=[parameter_min],
        c=:viridis,
        clims=(parameter_min, parameter_max),
        colorbar=true,
        colorbar_title=parameter_label,
        label="",
        markersize=0,
    )

    for (k, parameter_value) in enumerate(parameter_values)
        rows = filter(row -> getfield(row, parameter_name) == parameter_value, trajectory_rows)
        sort!(rows; by=row -> row.step)
        Rgs = [row.mean_gyration_radius for row in rows]
        Ms = [row.mean_particle_count for row in rows]
        Ms_std = [row.std_particle_count for row in rows]
        plot!(
            p1,
            Rgs,
            Ms;
            label="",
            color=colors[k],
            linewidth=1.5,
            marker=:none,
            ribbon=Ms_std,
            fillcolor=colors[k],
            fillalpha=0.10,
            colorbar_entry=false,
        )
    end

    if !isempty(dimension_rows)
        xs = [getfield(row, parameter_name) for row in dimension_rows]
        means = [row.mean_dimension for row in dimension_rows]
        stds = [row.std_dimension for row in dimension_rows]
        scatter!(
            p2,
            xs,
            means;
            yerror=stds,
            markersize=6,
        )
    end

    p = plot(
        p1,
        p2;
        layout=(1, 2),
        size=(1420, 620),
        dpi=300,
        margin=4Plots.mm,
    )
    root, ext = splitext(output_path)
    savefig_auto_folder(p1, root * "_Rg_vs_M" * ext)
    savefig_auto_folder(p2, root * "_D_vs_" * String(parameter_name) * ext)
    savefig_auto_folder(p, output_path)
    return p
end


function _rerender_dimension_plot_from_csv(;
    input_dir::String="plots/ass_2",
    output_path::String,
    prefix::String,
    parameter_name::Symbol,
)
    trajectory_rows = _read_eta_summary_csv(
        joinpath(input_dir, prefix * "_trajectory_summary.csv"),
        [
            parameter_name => Float64,
            :step => Int,
            :mean_particle_count => Float64,
            :std_particle_count => Float64,
            :mean_gyration_radius => Float64,
            :std_gyration_radius => Float64,
            :n_repeats => Int,
        ],
    )
    dimension_rows = _read_eta_summary_csv(
        joinpath(input_dir, prefix * "_dimension_summary.csv"),
        [
            parameter_name => Float64,
            :mean_dimension => Float64,
            :std_dimension => Float64,
            :n_valid_runs => Int,
        ],
    )
    return _plot_dimension_sweep_experiment(
        trajectory_rows,
        dimension_rows;
        output_path=output_path,
        parameter_name=parameter_name,
    )
end


function rerender_eta_dimension_plot_from_csv(;
    input_dir::String="plots/ass_2",
    output_path::String=joinpath(input_dir, "eta_sweep_M_vs_Rg_and_D.png"),
)
    return _rerender_dimension_plot_from_csv(
        input_dir=input_dir,
        output_path=output_path,
        prefix="eta_sweep",
        parameter_name=:eta,
    )
end


function rerender_ps_dimension_plot_from_csv(;
    input_dir::String="plots/ass_2",
    output_path::String=joinpath(input_dir, "p_s_sweep_M_vs_Rg_and_D.png"),
)
    return _rerender_dimension_plot_from_csv(
        input_dir=input_dir,
        output_path=output_path,
        prefix="p_s_sweep",
        parameter_name=:p_s,
    )
end


function plot_dla_eta_samples(;
    etas::AbstractVector{<:Real}=[0.5, 1.0, 2.0],
    N::Int=100,
    L::Float64=1.0,
    tol::Float64=1e-3,
    frames::Int=1000,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.91,
    backend::Symbol=:cpu,
    solver::Symbol=:rb_sor,
    mg_ncycles::Int=8,
    mg_levels::Int=0,
    mg_pre_sweeps::Int=2,
    mg_post_sweeps::Int=2,
    mg_coarse_sweeps::Int=30,
    mg_smoother::Symbol=:rb_sor,
    output_path::String="plots/ass_2/dla_eta_samples.png",
)
    length(etas) == 3 || throw(ArgumentError("etas must contain exactly 3 values for the 1x3 layout."))
    backend_resolved = _validate_backend(backend)
    fig, axs = PythonPlot.subplots(1, 3; figsize=(13.8, 4.8), constrained_layout=false)
    subplot_left = 0.06
    subplot_right = 0.92
    subplot_bottom = 0.13
    subplot_top = 0.88
    fig.subplots_adjust(left=subplot_left, right=subplot_right, bottom=subplot_bottom, top=subplot_top, wspace=0.18)
    images = Any[]

    for (idx, eta) in enumerate(etas)
        result = run_diffusion_limited_aggregation(
            N,
            L,
            tol,
            frames;
            i_max_conv=i_max_conv,
            omega_sor=omega_sor,
            solver=solver,
            backend=backend_resolved,
            eta=Float64(eta),
            candidate_picker=choose_candidate,
            mg_ncycles=mg_ncycles,
            mg_levels=mg_levels,
            mg_pre_sweeps=mg_pre_sweeps,
            mg_post_sweeps=mg_post_sweeps,
            mg_coarse_sweeps=mg_coarse_sweeps,
            mg_smoother=mg_smoother,
            collect_history=true,
            save_final_plot=false,
            verbose=false,
            do_gif=false,
        )

        c_plot = superimpose_c_sink(result.final_c, result.final_sink)
        ax = axs[idx-1]
        image = ax.imshow(
            permutedims(c_plot);
            origin="lower",
            extent=(0.0, L, 0.0, L),
            vmin=0.0,
            vmax=1.0,
            cmap="inferno",
            interpolation="nearest",
        )
        push!(images, image)

        ax.set_title(@sprintf("eta = %.1f", eta), fontsize=12, pad=8)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, L)
        ax.set_ylim(0.0, L)
        ax.set_xticks(collect(0.0:L/4:L))
        ax.set_yticks(collect(0.0:L/4:L))
        ax.tick_params(labelsize=9)
        if idx == 2
            ax.set_xlabel("x", fontsize=11)
        end
        if idx == 1
            ax.set_ylabel("y", fontsize=11)
        end
    end

    cax = fig.add_axes((0.935, subplot_bottom, 0.015, subplot_top - subplot_bottom))
    colorbar = fig.colorbar(images[end], cax=cax)
    colorbar.set_label("c", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    mkpath(dirname(output_path))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return fig
end


function plot_dla_ps_samples(;
    p_s_values::AbstractVector{<:Real}=[0.1, 0.5, 1.0],
    N::Int=100,
    L::Float64=1.0,
    tol::Float64=1e-3,
    frames::Int=1000,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.91,
    backend::Symbol=:cpu,
    solver::Symbol=:rb_sor,
    mc_i_max::Int=10_000_000,
    mg_ncycles::Int=8,
    mg_levels::Int=0,
    mg_pre_sweeps::Int=2,
    mg_post_sweeps::Int=2,
    mg_coarse_sweeps::Int=30,
    mg_smoother::Symbol=:rb_sor,
    output_path::String="plots/ass_2/dla_ps_samples.png",
)
    length(p_s_values) == 3 || throw(ArgumentError("p_s_values must contain exactly 3 values for the 1x3 layout."))
    all(0.0 .< collect(Float64, p_s_values) .<= 1.0) || throw(ArgumentError("All p_s values must satisfy 0 < p_s <= 1."))
    backend_resolved = _validate_backend(backend)
    fig, axs = PythonPlot.subplots(1, 3; figsize=(13.8, 4.8), constrained_layout=false)
    subplot_left = 0.06
    subplot_right = 0.92
    subplot_bottom = 0.13
    subplot_top = 0.88
    fig.subplots_adjust(left=subplot_left, right=subplot_right, bottom=subplot_bottom, top=subplot_top, wspace=0.18)
    images = Any[]

    for (idx, p_s) in enumerate(p_s_values)
        result = run_diffusion_limited_aggregation(
            N,
            L,
            tol,
            frames;
            i_max_conv=i_max_conv,
            omega_sor=omega_sor,
            solver=solver,
            backend=backend_resolved,
            p_s=Float64(p_s),
            candidate_picker=choose_candidate_monte_carlo,
            mc_i_max=mc_i_max,
            mg_ncycles=mg_ncycles,
            mg_levels=mg_levels,
            mg_pre_sweeps=mg_pre_sweeps,
            mg_post_sweeps=mg_post_sweeps,
            mg_coarse_sweeps=mg_coarse_sweeps,
            mg_smoother=mg_smoother,
            collect_history=true,
            save_final_plot=false,
            verbose=false,
            do_gif=false,
        )

        c_plot = superimpose_c_sink(result.final_c, result.final_sink)
        ax = axs[idx-1]
        image = ax.imshow(
            permutedims(c_plot);
            origin="lower",
            extent=(0.0, L, 0.0, L),
            vmin=0.0,
            vmax=1.0,
            cmap="inferno",
            interpolation="nearest",
        )
        push!(images, image)

        ax.set_title(@sprintf("p_s = %.1f", p_s), fontsize=12, pad=8)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, L)
        ax.set_ylim(0.0, L)
        ax.set_xticks(collect(0.0:L/4:L))
        ax.set_yticks(collect(0.0:L/4:L))
        ax.tick_params(labelsize=9)
        if idx == 2
            ax.set_xlabel("x", fontsize=11)
        end
        if idx == 1
            ax.set_ylabel("y", fontsize=11)
        end
    end

    cax = fig.add_axes((0.935, subplot_bottom, 0.015, subplot_top - subplot_bottom))
    colorbar = fig.colorbar(images[end], cax=cax)
    colorbar.set_label("c", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)

    mkpath(dirname(output_path))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return fig
end


function run_diffusion_limited_aggregation(
    N::Int,
    L::Float64,
    tol::Float64,
    frames::Int
    ;
    i_max_conv::Int=10_000,
    omega_sor::Float64,
    solver::Symbol=:rb_sor,
    backend::Symbol=:auto,
    eta::Union{Float64,Nothing}=nothing,
    p_s::Union{Float64,Nothing}=nothing,
    candidate_picker::Function=choose_candidate,
    mc_i_max::Int=1_000_000,
    mg_ncycles::Int=8,
    mg_levels::Int=0,
    mg_pre_sweeps::Int=2,
    mg_post_sweeps::Int=2,
    mg_coarse_sweeps::Int=30,
    mg_smoother::Symbol=:rb_sor,
    do_gif::Bool=false,
    collect_history::Bool=false,
    history_stride::Int=1,
    save_final_plot::Bool=true,
    verbose::Bool=true,
    plot_output_dir::String="plots")
    # Instantiate starting conditions

    # Source
    c_source = zeros(Bool, N, N)
    c_source[:, end] .= 1
    # Sink
    c_sink = zeros(Bool, N, N)
    # Single seed
    c_sink[N÷2, 1] = true
    # Start with equilibrium solution of initial conditions
    c = c_anal_2d(N)

    backend_resolved = _validate_backend(backend)

    c_old = nothing
    diffs = nothing

    if backend_resolved == :metal
        c_sink = MtlMatrix(c_sink)
        c = MtlMatrix(Matrix{Float32}(c))
        # Pre allocate GPU matrices
        c_old = similar(c)
        diffs = similar(c)
    elseif backend_resolved == :cuda
        c_sink = CuArray(c_sink)
        c = CuArray(Matrix{Float32}(c))
        c_old = similar(c)
        diffs = similar(c)
    end

    # Allocate once
    cpu_c = Matrix{Float64}(Array(c))
    cpu_sink = Matrix{Bool}(Array(c_sink))

    history_stride > 0 || throw(ArgumentError("history_stride must be > 0"))
    sink_history = collect_history ? Matrix{Bool}[copy(cpu_sink)] : Matrix{Bool}[]
    frame_indices = collect_history ? Int[0] : Int[]

    if verbose
        @info "Created initial conditions"
    end

    # Allocate plots vector
    plots = do_gif ? Vector{Plot{GRBackend}}(undef, frames) : Plot{GRBackend}[]
    # Fetch plotting kwargs
    heatmap_kwargs = get_heatmap_kwargs(N, L)

    if verbose
        @showprogress "Solving frames" for i in 1:1:frames
            diffusion_limited_aggregation_step!(
                c,
                c_sink,
                c_source,
                cpu_c,
                cpu_sink
                ;
                tol=tol,
                i_max_conv=i_max_conv,
                omega_sor=omega_sor,
                solver=solver,
                backend=backend_resolved,
                eta=eta,
                p_s=p_s,
                candidate_picker=candidate_picker,
                mc_i_max=mc_i_max,
                mg_ncycles=mg_ncycles,
                mg_levels=mg_levels,
                mg_pre_sweeps=mg_pre_sweeps,
                mg_post_sweeps=mg_post_sweeps,
                mg_coarse_sweeps=mg_coarse_sweeps,
                mg_smoother=mg_smoother,
                c_old=c_old,
                diffs=diffs
            )

            if do_gif
                plots[i] = plot_DLA_frame(cpu_c, cpu_sink; title=@sprintf("Iteration %03d", i), heatmap_kwargs...)
            end
            if collect_history && (i % history_stride == 0)
                push!(sink_history, copy(cpu_sink))
                push!(frame_indices, i)
            end
        end
    else
        for i in 1:1:frames
            diffusion_limited_aggregation_step!(
                c,
                c_sink,
                c_source,
                cpu_c,
                cpu_sink
                ;
                tol=tol,
                i_max_conv=i_max_conv,
                omega_sor=omega_sor,
                solver=solver,
                backend=backend_resolved,
                eta=eta,
                p_s=p_s,
                candidate_picker=candidate_picker,
                mc_i_max=mc_i_max,
                mg_ncycles=mg_ncycles,
                mg_levels=mg_levels,
                mg_pre_sweeps=mg_pre_sweeps,
                mg_post_sweeps=mg_post_sweeps,
                mg_coarse_sweeps=mg_coarse_sweeps,
                mg_smoother=mg_smoother,
                c_old=c_old,
                diffs=diffs
            )

            if do_gif
                plots[i] = plot_DLA_frame(cpu_c, cpu_sink; title=@sprintf("Iteration %03d", i), heatmap_kwargs...)
            end
            if collect_history && (i % history_stride == 0)
                push!(sink_history, copy(cpu_sink))
                push!(frame_indices, i)
            end
        end
    end

    if save_final_plot
        filename_final_state = joinpath(plot_output_dir, "diffusion_limited_aggregation_end_N=$(N)_$(candidate_picker).png")
        p_final = plot_DLA_frame(cpu_c, cpu_sink; title="Final Frame", heatmap_kwargs...)
        savefig_auto_folder(p_final, filename_final_state)
        if verbose
            @info "Saved final state to $filename_final_state"
        end
    end


    # Save gif of the process
    if do_gif
        filename_gif = joinpath(plot_output_dir, "diffusion_limited_aggregation_N=$(N)_$(candidate_picker).gif")
        @time "Saved gif to $filename_gif" begin
            distributed_gif(plots, filename_gif; fps=60, do_palette=true, width=900)
        end
    end

    if collect_history
        return (
            sink_history=sink_history,
            frame_indices=frame_indices,
            final_c=copy(cpu_c),
            final_sink=copy(cpu_sink),
            metadata=(
                N=N,
                L=L,
                eta=eta,
                frames=frames,
                backend=backend_resolved,
                solver=solver,
                history_stride=history_stride,
            ),
        )
    end

    return nothing
end


function _synchronize_backend(backend::Symbol)
    if backend == :metal
        Metal.synchronize()
    elseif backend == :cuda
        CUDA.synchronize()
    end
    return nothing
end


function _time_one_dla_run(;
    N::Int,
    L::Float64,
    tol::Float64,
    frames::Int,
    i_max_conv::Int,
    omega_sor::Float64,
    solver::Symbol,
    backend::Symbol,
    eta::Union{Float64,Nothing},
    mg_ncycles::Int,
    mg_levels::Int,
    mg_pre_sweeps::Int,
    mg_post_sweeps::Int,
    mg_coarse_sweeps::Int,
    mg_smoother::Symbol,
)::Float64
    t0 = time_ns()
    run_diffusion_limited_aggregation(
        N,
        L,
        tol,
        frames;
        i_max_conv=i_max_conv,
        omega_sor=omega_sor,
        solver=solver,
        backend=backend,
        eta=eta,
        candidate_picker=choose_candidate,
        mg_ncycles=mg_ncycles,
        mg_levels=mg_levels,
        mg_pre_sweeps=mg_pre_sweeps,
        mg_post_sweeps=mg_post_sweeps,
        mg_coarse_sweeps=mg_coarse_sweeps,
        mg_smoother=mg_smoother,
        do_gif=false,
        save_final_plot=false,
        verbose=false,
    )
    _synchronize_backend(backend)
    return (time_ns() - t0) / 1e9
end


function _aggregate_timing_results(raw_rows::Vector{NamedTuple})
    grouped = Dict{Tuple{Int,Symbol,Symbol},Vector{Float64}}()
    for row in raw_rows
        key = (row.N, row.backend, row.solver)
        if !haskey(grouped, key)
            grouped[key] = Float64[]
        end
        push!(grouped[key], row.time_s)
    end

    summary = NamedTuple[]
    keys_sorted = sort!(collect(keys(grouped)); by=x -> (x[2], x[3], x[1]))
    for (N, backend, solver) in keys_sorted
        ts = grouped[(N, backend, solver)]
        push!(summary, (
            N=N,
            backend=backend,
            solver=solver,
            mean_time_s=mean(ts),
            std_time_s=std(ts),
            n_repeats=length(ts),
        ))
    end
    return summary
end


function _write_timing_csv(path::String, rows::Vector{NamedTuple}; summary::Bool=false)
    mkpath(dirname(path))
    open(path, "w") do io
        if summary
            println(io, "N,backend,solver,mean_time_s,std_time_s,n_repeats")
            for r in rows
                println(io, "$(r.N),$(r.backend),$(r.solver),$(r.mean_time_s),$(r.std_time_s),$(r.n_repeats)")
            end
        else
            println(io, "N,backend,solver,repeat,time_s")
            for r in rows
                println(io, "$(r.N),$(r.backend),$(r.solver),$(r.repeat),$(r.time_s)")
            end
        end
    end
    return path
end


function _plot_dla_scaling(summary_rows::Vector{NamedTuple}; output_path::String)
    combos = unique((r.backend, r.solver) for r in summary_rows)
    p = plot(
        xlabel="N",
        ylabel="Computation time (s)",
        title="DLA Runtime Scaling",
        dpi=200,
    )
    for (backend, solver) in combos
        rows = filter(r -> r.backend == backend && r.solver == solver, summary_rows)
        sort!(rows; by=r -> r.N)
        Ns = [r.N for r in rows]
        means = [r.mean_time_s for r in rows]
        stds = [r.std_time_s for r in rows]
        plot!(p, Ns, means; ribbon=stds, marker=:circle, label="$(backend)+$(solver)")
    end
    savefig_auto_folder(p, output_path)
    return p
end


function run_dla_scaling_experiment(;
    N_values=40:20:200,
    repeats::Int=20,
    L::Float64=1.0,
    tol::Float64=1e-3,
    frames::Int=200,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.91,
    eta::Union{Float64,Nothing}=1.5,
    backend::Symbol=:auto,
    mg_ncycles::Int=8,
    mg_levels::Int=0,
    mg_pre_sweeps::Int=2,
    mg_post_sweeps::Int=2,
    mg_coarse_sweeps::Int=30,
    mg_smoother::Symbol=:rb_sor,
    output_dir::String="plots/ass_2",
    save_csv::Bool=true,
    seed_base::Int=1234,
)
    Ns = collect(Int, N_values)
    combos = _valid_solver_backend_pairs(; backend=backend)
    @info "Running DLA scaling experiment for $(length(Ns)) grid sizes and $(length(combos)) solver/backend combinations."

    raw_rows = NamedTuple[]
    for combo in combos
        for N in Ns
            # Warm-up run for JIT/runtime startup effects.
            _ = _time_one_dla_run(
                N=N, L=L, tol=tol, frames=frames, i_max_conv=i_max_conv, omega_sor=omega_sor,
                solver=combo.solver, backend=combo.backend, eta=eta,
                mg_ncycles=mg_ncycles, mg_levels=mg_levels, mg_pre_sweeps=mg_pre_sweeps,
                mg_post_sweeps=mg_post_sweeps, mg_coarse_sweeps=mg_coarse_sweeps, mg_smoother=mg_smoother,
            )

            @info "Timing N=$N backend=$(combo.backend) solver=$(combo.solver)"
            for rep in 1:repeats
                seed!(seed_base + hash((N, combo.backend, combo.solver, rep)))
                t_s = _time_one_dla_run(
                    N=N, L=L, tol=tol, frames=frames, i_max_conv=i_max_conv, omega_sor=omega_sor,
                    solver=combo.solver, backend=combo.backend, eta=eta,
                    mg_ncycles=mg_ncycles, mg_levels=mg_levels, mg_pre_sweeps=mg_pre_sweeps,
                    mg_post_sweeps=mg_post_sweeps, mg_coarse_sweeps=mg_coarse_sweeps, mg_smoother=mg_smoother,
                )
                push!(raw_rows, (N=N, backend=combo.backend, solver=combo.solver, repeat=rep, time_s=t_s))
            end
        end
    end

    summary_rows = _aggregate_timing_results(raw_rows)

    if save_csv
        _write_timing_csv(joinpath(output_dir, "dla_scaling_raw.csv"), raw_rows; summary=false)
        _write_timing_csv(joinpath(output_dir, "dla_scaling_summary.csv"), summary_rows; summary=true)
    end

    plot_path = joinpath(output_dir, "dla_scaling_time_vs_N.png")
    p = _plot_dla_scaling(summary_rows; output_path=plot_path)
    @info "Saved scaling plot to $plot_path"

    return (raw=raw_rows, summary=summary_rows, plot=p, combos=combos)
end


function _run_dla_dimension_experiment(;
    N::Int=100,
    L::Float64=1.0,
    tol::Float64=1e-3,
    frames::Int=500,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.91,
    parameter_name::Symbol,
    parameter_values,
    repeats::Int=30,
    backend::Symbol=:auto,
    solver::Symbol=:rb_sor,
    candidate_picker::Function,
    mc_i_max::Int=1_000_000,
    mg_ncycles::Int=8,
    mg_levels::Int=0,
    mg_pre_sweeps::Int=2,
    mg_post_sweeps::Int=2,
    mg_coarse_sweeps::Int=30,
    mg_smoother::Symbol=:rb_sor,
    output_dir::String="plots/ass_2",
    save_csv::Bool=true,
    seed_base::Int=1234,
    history_stride::Int=1,
    analysis_label::String,
)
    repeats > 0 || throw(ArgumentError("repeats must be > 0"))
    history_stride > 0 || throw(ArgumentError("history_stride must be > 0"))
    backend_resolved = _default_eta_backend(backend)
    values = collect(Float64, parameter_values)
    isempty(values) && throw(ArgumentError("parameter_values must not be empty"))

    raw_rows = NamedTuple[]
    trajectory_rows = NamedTuple[]
    dimension_rows = NamedTuple[]
    invalid_fit_runs = 0

    for (parameter_idx, parameter_value) in enumerate(values)
        if repeats > 0
            @info "Running $(analysis_label) for $(parameter_name)=$(parameter_value) ($parameter_idx/$(length(values)))"
        end

        Mss = Vector{Vector{Float64}}()
        Rgss = Vector{Vector{Float64}}()
        Ds = Float64[]
        frame_template = Int[]

        for rep in 1:repeats
            seed!(seed_base + hash((parameter_name, parameter_value, rep, N, backend_resolved, solver)))
            eta_arg = parameter_name == :eta ? parameter_value : nothing
            p_s_arg = parameter_name == :p_s ? parameter_value : nothing
            result = run_diffusion_limited_aggregation(
                N,
                L,
                tol,
                frames;
                i_max_conv=i_max_conv,
                omega_sor=omega_sor,
                solver=solver,
                backend=backend_resolved,
                eta=eta_arg,
                p_s=p_s_arg,
                candidate_picker=candidate_picker,
                mc_i_max=mc_i_max,
                mg_ncycles=mg_ncycles,
                mg_levels=mg_levels,
                mg_pre_sweeps=mg_pre_sweeps,
                mg_post_sweeps=mg_post_sweeps,
                mg_coarse_sweeps=mg_coarse_sweeps,
                mg_smoother=mg_smoother,
                do_gif=false,
                collect_history=true,
                history_stride=history_stride,
                save_final_plot=false,
                verbose=false,
            )

            Ms = Float64[]
            Rgs = Float64[]
            for (step, mask) in zip(result.frame_indices, result.sink_history)
                n_particles, r_g = cluster_stats(mask; occupied_value=true, normalized=false)
                push!(Ms, Float64(n_particles))
                push!(Rgs, r_g)
                push!(raw_rows, (
                    parameter_name => parameter_value,
                    repeat=rep,
                    step=step,
                    particle_count=Float64(n_particles),
                    gyration_radius=r_g,
                ))
            end
            if isempty(frame_template)
                frame_template = copy(result.frame_indices)
            end

            valid = isfinite.(Rgs) .& isfinite.(Ms) .& (Rgs .> 0.0) .& (Ms .> 1.0)
            x_valid = log.(Rgs[valid])
            y_valid = log.(Ms[valid])
            slope = _linear_fit_slope(x_valid, y_valid)
            if isfinite(slope)
                push!(Ds, slope)
            else
                invalid_fit_runs += 1
            end

            push!(Mss, Ms)
            push!(Rgss, Rgs)
        end

        common_len = minimum(length.(Mss))
        if common_len < 2
            @warn "Skipping $(parameter_name)=$(parameter_value) due to too-short trajectories." common_len
            continue
        end

        Mmat = reduce(hcat, [series[1:common_len] for series in Mss])
        Rgmat = reduce(hcat, [series[1:common_len] for series in Rgss])
        Ms_mean = vec(mean(Mmat; dims=2))
        Ms_std = vec(std(Mmat; dims=2))
        Rgs_mean = vec(mean(Rgmat; dims=2))
        Rgs_std = vec(std(Rgmat; dims=2))
        steps = frame_template[1:common_len]

        for idx in 1:common_len
            push!(trajectory_rows, (
                parameter_name => parameter_value,
                step=steps[idx],
                mean_particle_count=Ms_mean[idx],
                std_particle_count=Ms_std[idx],
                mean_gyration_radius=Rgs_mean[idx],
                std_gyration_radius=Rgs_std[idx],
                n_repeats=repeats,
            ))
        end

        d_mean = isempty(Ds) ? NaN : mean(Ds)
        d_std = length(Ds) > 1 ? std(Ds) : 0.0
        push!(dimension_rows, (
            parameter_name => parameter_value,
            mean_dimension=d_mean,
            std_dimension=d_std,
            n_valid_runs=length(Ds),
        ))
    end

    if invalid_fit_runs > 0
        @warn "Skipped invalid fractal-dimension fits." invalid_fit_runs
    end

    save_csv && _write_dimension_sweep_outputs(output_dir, analysis_label, parameter_name, raw_rows, trajectory_rows, dimension_rows)

    plot_path = joinpath(output_dir, analysis_label * "_M_vs_Rg_and_D.png")
    p = _plot_dimension_sweep_experiment(
        trajectory_rows,
        dimension_rows;
        output_path=plot_path,
        parameter_name=parameter_name,
    )
    @info "Saved $(analysis_label) plot to $plot_path"

    return (raw=raw_rows, trajectories=trajectory_rows, dimensions=dimension_rows, plot=p)
end


function run_eta_dimension_experiment(;
    N::Int=100,
    L::Float64=1.0,
    tol::Float64=1e-3,
    frames::Int=500,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.91,
    eta_values=0.1:0.1:2.0,
    repeats::Int=30,
    backend::Symbol=:auto,
    solver::Symbol=:rb_sor,
    mg_ncycles::Int=8,
    mg_levels::Int=0,
    mg_pre_sweeps::Int=2,
    mg_post_sweeps::Int=2,
    mg_coarse_sweeps::Int=30,
    mg_smoother::Symbol=:rb_sor,
    output_dir::String="plots/ass_2",
    save_csv::Bool=true,
    seed_base::Int=1234,
    history_stride::Int=1,
)
    return _run_dla_dimension_experiment(
        N=N,
        L=L,
        tol=tol,
        frames=frames,
        i_max_conv=i_max_conv,
        omega_sor=omega_sor,
        parameter_name=:eta,
        parameter_values=eta_values,
        repeats=repeats,
        backend=backend,
        solver=solver,
        candidate_picker=choose_candidate,
        mc_i_max=1_000_000,
        mg_ncycles=mg_ncycles,
        mg_levels=mg_levels,
        mg_pre_sweeps=mg_pre_sweeps,
        mg_post_sweeps=mg_post_sweeps,
        mg_coarse_sweeps=mg_coarse_sweeps,
        mg_smoother=mg_smoother,
        output_dir=output_dir,
        save_csv=save_csv,
        seed_base=seed_base,
        history_stride=history_stride,
        analysis_label="eta_sweep",
    )
end


function run_ps_dimension_experiment(;
    N::Int=100,
    L::Float64=1.0,
    tol::Float64=1e-3,
    frames::Int=500,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.91,
    p_s_values=0.1:0.1:1.0,
    repeats::Int=30,
    backend::Symbol=:auto,
    solver::Symbol=:rb_sor,
    mg_ncycles::Int=8,
    mg_levels::Int=0,
    mg_pre_sweeps::Int=2,
    mg_post_sweeps::Int=2,
    mg_coarse_sweeps::Int=30,
    mg_smoother::Symbol=:rb_sor,
    output_dir::String="plots/ass_2",
    save_csv::Bool=true,
    seed_base::Int=1234,
    history_stride::Int=1,
    mc_i_max::Int=10_000_000,
)
    all(0.0 .< collect(Float64, p_s_values) .<= 1.0) || throw(ArgumentError("All p_s values must satisfy 0 < p_s <= 1."))
    return _run_dla_dimension_experiment(
        N=N,
        L=L,
        tol=tol,
        frames=frames,
        i_max_conv=i_max_conv,
        omega_sor=omega_sor,
        parameter_name=:p_s,
        parameter_values=p_s_values,
        repeats=repeats,
        backend=backend,
        solver=solver,
        candidate_picker=choose_candidate_monte_carlo,
        mc_i_max=mc_i_max,
        mg_ncycles=mg_ncycles,
        mg_levels=mg_levels,
        mg_pre_sweeps=mg_pre_sweeps,
        mg_post_sweeps=mg_post_sweeps,
        mg_coarse_sweeps=mg_coarse_sweeps,
        mg_smoother=mg_smoother,
        output_dir=output_dir,
        save_csv=save_csv,
        seed_base=seed_base,
        history_stride=history_stride,
        analysis_label="p_s_sweep",
    )
end

end # module
