module Plotting

using Plots
using LinearAlgebra
using Statistics
using FileIO

default(fontfamily="Computer Modern")

import ..Model: analytical_profile_series
import ..DataIO: load_output

export plot_animation, plot_profiles, plot_2d_concentration, plot_wave_final, animate_wave_all, plot_steadystate, plot_concentration_profiles_steady, plot_convergence_its, plot_omega_optimisation, plot_omega_sweep_panels

function _sink_mask(
    dims::Tuple{Int,Int},
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}},
)
    if isnothing(sink_indices)
        return nothing
    end
    mask = falses(dims...)
    mask[sink_indices] .= true
    return mask
end

function _overlay_sink_outline!(
    p,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    sink_mask::Union{Nothing,BitMatrix},
)
    if isnothing(sink_mask) || !any(sink_mask)
        return nothing
    end

    contour!(
        p,
        x,
        y,
        Float64.(sink_mask)',
        levels=[0.5],
        color=:white,
        linewidth=2,
        label=false,
    )
    return nothing
end

function _overlay_sink_silly!(
    p,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    sink_mask::Union{Nothing,BitMatrix},
    silly::Bool,
    silly_image_path::String,
)
    if !silly || isnothing(sink_mask) || !any(sink_mask)
        return nothing
    end

    img_path = isabspath(silly_image_path) ? silly_image_path : normpath(joinpath(@__DIR__, silly_image_path))
    if !isfile(img_path)
        @warn "Silly image not found, skipping overlay." silly_image_path img_path
        return nothing
    end

    img = try
        load(img_path)
    catch err
        @warn "Failed to load silly image, skipping overlay." silly_image_path img_path exception = (err, catch_backtrace())
        return nothing
    end
    img2d = ndims(img) == 2 ? img : img[:, :, 1]

    rows = findall(vec(any(sink_mask, dims=2)))
    cols = findall(vec(any(sink_mask, dims=1)))
    i_min, i_max = minimum(rows), maximum(rows)
    j_min, j_max = minimum(cols), maximum(cols)

    plot!(
        p,
        [x[i_min], x[i_max]],
        [y[j_min], y[j_max]],
        img2d;
        seriestype=:image,
        label=false,
    )
    return nothing
end

function _save_plot(p, outfile::String)
    ext = lowercase(splitext(outfile)[2])
    if ext == ".png"
        png(p, outfile)
    else
        savefig(p, outfile)
    end
    return outfile
end

function plot_animation(
    filepath::String="output/data/output.h5";
    fps::Int=30,
    filename::String="output/img/diffusion_anim.mp4",
    max_frames::Int=300,
    stride::Union{Int,Nothing}=nothing,
)
    """
    Loads diffusion data and creates an animated heatmap over time.
    """
    c, dx, dy, steps, _ = load_output(filepath)
    nt = size(c, 3)

    x = (0:size(c, 1)-1) .* dx
    y = (0:size(c, 2)-1) .* dy

    local_stride = stride === nothing ? max(1, cld(nt, max_frames)) : stride
    frame_indices = 1:local_stride:nt

    anim = @animate for t in frame_indices
        heatmap(
            x,
            y,
            c[:, :, t]',
            xlabel="x",
            ylabel="y",
            title="Diffusion 2D - step $(steps[t])",
            aspect_ratio=1,
            color=:viridis,
            clims=(0, 1),
            xlims=(x[1], x[end]),
            ylims=(y[1], y[end]),
        )
    end

    mkpath(dirname(filename))
    mp4(anim, filename, fps=fps)
    return nothing
end

function plot_profiles(
    filepath::String="output/data/output.h5",
    output::String="output/img/profiles.png";
    times::Vector{Float64}=Float64[],
    x_index::Int=0,
    average_x::Bool=true,
    D::Float64=1.0,
    n_terms::Int=200,
)
    """
    Plots concentration as a function of y for selected time points.
    """
    c, _, dy, _, tvals = load_output(filepath)
    y = (0:size(c, 2)-1) .* dy
    L = y[end]

    c3 = ndims(c) == 2 ? reshape(c, size(c, 1), size(c, 2), 1) : c
    nt = size(c3, 3)
    has_time = tvals !== nothing

    if isempty(times)
        indices = collect(1:nt)
    else
        if has_time
            indices = [findmin(abs.(tvals .- t))[2] for t in times]
        else
            indices = Int.(times)
        end
    end

    p = plot(dpi=300)
    for (k, idx) in enumerate(indices)
        slice = c3[:, :, idx]
        prof = average_x ? vec(mean(slice, dims=1)) : slice[x_index == 0 ? 1 : x_index, :]
        label = has_time ? "t=$(tvals[idx])" : "frame=$idx"
        plot!(p, y, prof, label=label, seriescolor=k)

        if has_time
            t = tvals[idx]
            if t > 0
                c_analytical = analytical_profile_series(y, t, D, L; n_terms=n_terms)
                plot!(
                    p,
                    y,
                    c_analytical,
                    linestyle=:dash,
                    label="analytical t=$(round(t; digits=4))",
                    seriescolor=k,
                )
            end
        end
    end

    if has_time && !isempty(indices)
        final_idx = indices[end]
        final_t = tvals[final_idx]

        if final_t > 0
            numerical_final = average_x ? vec(mean(c3[:, :, final_idx], dims=1)) : c3[x_index == 0 ? 1 : x_index, :, final_idx]
            analytical_final = analytical_profile_series(y, final_t, D, L; n_terms=n_terms)
            err = norm(numerical_final - analytical_final) / norm(analytical_final)
            println("Relative L2 error at final time (t=$(final_t)): $(round(err; sigdigits=3))")
        end
    end

    xlabel!(p, "y")
    ylabel!(p, "concentration")
    title!(p, "Concentration vs y")
    display(p)
    mkpath(dirname(output))
    savefig(output)

    return nothing
end

function plot_2d_concentration(
    filepath::String="output/data/output.h5",
    output::String="output/img/concentration.png",
    timesteps::Vector{Float64}=[1.0],
)
    """
    Plots 2D heatmaps of concentration for requested times.
    """
    c, dx, dy, _, tvals = load_output(filepath)
    if tvals === nothing
        error("No time dataset found in $filepath")
    end

    x = (0:size(c, 1)-1) .* dx
    y = (0:size(c, 2)-1) .* dy

    timesteps_idx = [findmin(abs.(tvals .- t))[2] for t in timesteps]

    subplots = Any[]
    for (i, time_index) in enumerate(timesteps_idx)
        slice = c[:, :, time_index]
        t = timesteps[i]
        p = heatmap(
            x,
            y,
            slice',
            xlabel="x",
            ylabel="y",
            title="Concentration at time $t",
            aspect_ratio=1,
            color=:viridis,
        )

        push!(subplots, p)
    end

    p = plot(
        subplots...,
        layout=(ceil(Int, length(subplots) / 2), 2),
        size=(1800, 1200),
        dpi=300,
    )
    outfile = abspath(output)
    mkpath(dirname(outfile))
    _save_plot(p, outfile)

    return outfile
end

function plot_wave_final(
    psiss::Vector{<:AbstractMatrix},
    x::AbstractVector{<:Real},
    title_text::String;
    output::String="output/img/figure_1A.png",
)
    """
    Plots final-time wave profiles for multiple initial conditions.
    """
    p = plot(dpi=300)
    for (i, psis) in enumerate(psiss)
        plot!(p, x, psis[:, end], label="\\Psi_$i")
    end
    xlabel!(p, "x")
    ylabel!(p, "Psi")
    title!(p, title_text)
    mkpath(dirname(output))
    savefig(p, output)
    return nothing
end

function animate_wave_all(
    psiss::Vector{<:AbstractMatrix},
    x::AbstractVector{<:Real};
    fps::Int=30,
    ylim::Tuple{Real,Real}=(-1, 1),
    filename::String="output/img/animation_1C.mp4",
)
    """
    Creates a wave animation from multiple simulated profiles.
    """
    nt = size(psiss[1], 2)
    anim = @animate for n in 1:nt
        p = plot(ylim=ylim, legend=:bottom, size=(1200, 800), show=false)
        for (i, psis) in enumerate(psiss)
            plot!(p, x, psis[:, n], label="\\Psi_$i", show=false)
        end
        xlabel!(p, "x")
        ylabel!(p, "\\Psi")
        title!(p, "1D wave equation - timestep $n")
    end
    mkpath(dirname(filename))
    mp4(anim, filename, fps=fps)
    return nothing
end

function plot_steadystate(
    c::Matrix{Float64},
    output::String="output/img/steadystate.png";
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}}=nothing,
    silly_image_path::String="input/sink.png",
    silly::Bool=false,
)
    """
    Plots the steady state concentration profile as a 2D heatmap.
    """
    x = range(0, stop=1, length=size(c, 1))
    y = range(0, stop=1, length=size(c, 2))

    p = heatmap(
        x,
        y,
        c',
        xlabel="x",
        ylabel="y",
        title="Steady State Concentration Profile",
        aspect_ratio=1,
        color=:viridis,
        dpi=300
    )
    sink_mask = _sink_mask(size(c), sink_indices)
    if sink_indices !== nothing
        _overlay_sink_outline!(p, x, y, sink_mask)

        if silly
            _overlay_sink_silly!(p, x, y, sink_mask, silly, silly_image_path)
        end
    end

    outfile = abspath(output)
    mkpath(dirname(outfile))
    _save_plot(p, outfile)
    return outfile
end

function plot_concentration_profiles_steady(
    cs::Vector{Matrix{Float64}},
    methods::Vector{String}=String[],
    output::String="output/img/steadystate_profiles.png",
)
    """
    1. Plots concentration profiles along y for steady-state solutions and compares
    against an analytical profile on the same y-grid.
    2. Plots difference of methods against analytical solution in second subplot
    """
    if isempty(cs)
        error("cs must contain at least one solution matrix")
    end
    if !isempty(methods) && length(methods) != length(cs)
        error("methods length ($(length(methods))) must match number of solutions ($(length(cs)))")
    end

    y = range(0, stop=1, length=size(cs[1], 2))
    analytical_sol = analytical_profile_series(y, 1.0, 1.0, 1.0; n_terms=200)

    p = plot(dpi=300, title="Steady State Concentration Profiles", xlabel="y", ylabel="Concentration")
    plot!(p, y, analytical_sol, label="Analytical Solution", linestyle=:dash, color=:black)

    cmap = cgrad(:rainbow, length(cs), categorical=true)

    diffs = []
    for (i, c) in enumerate(cs)
        if size(c, 2) != length(y)
            error("All matrices in cs must have the same y-size")
        end
        profile = mean(c, dims=1) |> vec
        diff = profile - analytical_sol
        push!(diffs, diff)
        label = isempty(methods) ? "Numerical $i" : methods[i]
        plot!(p, y, profile, label=label, color=cmap[i])
    end

    # plot diff
    p2 = plot(dpi=300, title="Error vs Analytical Solution", xlabel="y", ylabel="Error", yscale=:log10)
    # yaxis log scale
    for (i, diff) in enumerate(diffs)
        err = max.(abs.(diff), eps(Float64))
        plot!(p2, y, err, label=isempty(methods) ? "$i" : methods[i], color=cmap[i])
    end

    p = plot(p, p2, layout=(2, 1), size=(1200, 800))

    mkpath(dirname(output))
    savefig(p, output)
end

function plot_convergence_its(
    deltas::Vector{Vector{Float64}},
    methods::Vector{String}=String[],
    output::String="output/img/steadystate_convergence.png",
)
    """
    Plots the number of iterations to convergence for different methods.
    """

    p = plot(dpi=300, yscale=:log10, title="Steady-state convergence", legend=:topright)
    for (i, delta) in enumerate(deltas)
        label = isempty(methods) ? "Method $i" : methods[i]
        plot!(p, 1:length(delta), delta, label=label)
    end
    xlabel!(p, "iteration")
    ylabel!(p, "\\delta")
    savefig(p, output)

end

function plot_omega_optimisation(
    omegas::AbstractVector{Float64},
    iterations::AbstractVector{<:Integer},
    output::String="output/img/optimise_omega.png",
    max_iters::Union{Nothing,Int}=nothing,
    converged::Union{Nothing,AbstractVector{Bool}}=nothing,
    computed::Union{Nothing,AbstractVector{Bool}}=nothing,
)
    """
    Plots the number of iterations to convergence for different omega values in SOR.
    """
    if length(omegas) != length(iterations)
        error("omegas and iterations must have the same length")
    end
    if !isnothing(converged) && length(converged) != length(iterations)
        error("converged must have the same length as iterations")
    end
    if !isnothing(computed) && length(computed) != length(iterations)
        error("computed must have the same length as iterations")
    end

    p = plot(
        xlabel="omega",
        ylabel="iterations to converge",
        yscale=:log10,
        title="Optimising omega for SOR",
        dpi=300,
        legend=:topleft,
    )

    comp_mask = isnothing(computed) ? trues(length(iterations)) : collect(computed)
    positive_mask = iterations .> 0

    if isnothing(max_iters)
        valid = comp_mask .& positive_mask
        if any(valid)
            plot!(p, omegas[valid], iterations[valid], marker=:circle, label="all runs")
        end
    else
        conv_mask = isnothing(converged) ? (iterations .< max_iters) : collect(converged)
        converged_valid = comp_mask .& conv_mask .& positive_mask
        capped = comp_mask .& .!conv_mask

        if any(converged_valid)
            plot!(p, omegas[converged_valid], iterations[converged_valid], marker=:circle, label="converged")
            local_omegas = omegas[converged_valid]
            local_iterations = iterations[converged_valid]
            i_best = argmin(local_iterations)
            omega_best = local_omegas[i_best]
            vline!(p, [omega_best], linestyle=:dash, color=:black, label="best ω=$(round(omega_best; digits=3))")
        end

        if any(capped)
            scatter!(
                p,
                omegas[capped],
                fill(max_iters, count(capped)),
                marker=:x,
                color=:red,
                label="capped at max_iters",
            )
        end
    end

    mkpath(dirname(output))
    savefig(p, output)

end

function plot_omega_optimisation(
    omegas::AbstractVector{Float64},
    Ns::AbstractVector{Int},
    iterations::AbstractMatrix{<:Integer},
    output::String="output/img/optimise_omega.png",
    max_iters::Union{Nothing,Int}=nothing,
    converged::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    computed::Union{Nothing,AbstractMatrix{Bool}}=nothing,
)
    """
    2D omega-optimisation plot for multiple grid sizes (one curve per N).

    Expected `iterations` shape is (length(omegas), length(Ns)).
    If transposed, it will be auto-corrected.
    """
    nW = length(omegas)
    nN = length(Ns)
    itmat = iterations
    conv = converged
    comp = computed

    if size(itmat) == (nN, nW)
        itmat = permutedims(itmat)
        if !isnothing(conv)
            conv = permutedims(conv)
        end
        if !isnothing(comp)
            comp = permutedims(comp)
        end
    elseif size(itmat) != (nW, nN)
        error("iterations must have shape (length(omegas), length(Ns)) or its transpose")
    end
    if !isnothing(conv) && size(conv) != size(itmat)
        error("converged must have same shape as iterations (or transposed together)")
    end
    if !isnothing(comp) && size(comp) != size(itmat)
        error("computed must have same shape as iterations (or transposed together)")
    end

    p = plot(
        xlabel="omega",
        ylabel="iterations to converge",
        yscale=:log10,
        title="Optimising omega for SOR",
        dpi=300,
        legend=:outerright,
    )

    colors = palette(:viridis, nN)
    for (j, N) in enumerate(Ns)
        vals = vec(itmat[:, j])
        comp_col = isnothing(comp) ? trues(length(vals)) : vec(comp[:, j])
        label = "N=$N"
        if isnothing(max_iters)
            plot!(p, omegas[comp_col], vals[comp_col], marker=:circle, color=colors[j], label=label)
        else
            conv_col = isnothing(conv) ? (vals .< max_iters) : vec(conv[:, j])
            capped = comp_col .& .!conv_col

            if any(conv_col)
                plot!(p, omegas[comp_col.&conv_col], vals[comp_col.&conv_col], marker=:circle, color=colors[j], label=label)
            end
            if any(capped)
                scatter!(
                    p,
                    omegas[capped],
                    fill(max_iters, count(capped)),
                    marker=:x,
                    color=colors[j],
                    label="$label (capped)",
                )
            end
        end
    end

    mkpath(dirname(output))
    savefig(p, output)
end

function plot_omega_sweep_panels(
    omegas::AbstractVector{Float64},
    Ns::AbstractVector{Int},
    iterations::AbstractMatrix{<:Real},
    output::String="output/img/omega_sweep_panels.png",
    max_iters::Union{Nothing,Int}=nothing,
    converged::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    computed::Union{Nothing,AbstractMatrix{Bool}}=nothing,
)
    """
    Two-panel visualization of omega-N sweep:
    1) Heatmap of iterations over (omega, N)
    2) Best converged omega(N) data points with theoretical reference curve
    """
    nN = length(Ns)
    nW = length(omegas)
    itmat = iterations
    conv = converged
    comp = computed

    if size(itmat) == (nW, nN)
        itmat = permutedims(itmat)
        if !isnothing(conv)
            conv = permutedims(conv)
        end
        if !isnothing(comp)
            comp = permutedims(comp)
        end
    elseif size(itmat) != (nN, nW)
        error("iterations must have shape (length(Ns), length(omegas)) or its transpose")
    end
    if !isnothing(conv) && size(conv) != size(itmat)
        error("converged must have same shape as iterations (or transposed together)")
    end
    if !isnothing(comp) && size(comp) != size(itmat)
        error("computed must have same shape as iterations (or transposed together)")
    end

    zvals = Float64.(itmat)
    comp_eff = isnothing(comp) ? trues(size(zvals)) : comp
    conv_eff = isnothing(conv) ? (isnothing(max_iters) ? trues(size(zvals)) : zvals .< max_iters) : conv

    z_heat = copy(zvals)
    z_heat[.!comp_eff] .= NaN
    z_heat[.!conv_eff] .= NaN
    if !isnothing(max_iters)
        z_heat = min.(z_heat, Float64(max_iters))
    end

    finite_vals = filter(isfinite, vec(z_heat))
    clims = isempty(finite_vals) ? (0.0, 1.0) : (minimum(finite_vals), maximum(finite_vals))

    p1 = heatmap(
        collect(omegas),
        Float64.(Ns),
        z_heat;
        xlabel="\\omega",
        ylabel="N",
        title="Iterations Heatmap (converged region)",
        color=:viridis,
        colorbar_title="iterations",
        clims=clims,
        left_margin=10Plots.mm,
        right_margin=14Plots.mm,
        bottom_margin=8Plots.mm,
        top_margin=6Plots.mm,
        titlefontsize=10,
        guidefontsize=10,
        tickfontsize=8,
        dpi=300,
    )

    x_nc = Float64[]
    y_nc = Float64[]
    for (iN, N) in enumerate(Ns), (iW, ω) in enumerate(omegas)
        if comp_eff[iN, iW] && !conv_eff[iN, iW]
            push!(x_nc, ω)
            push!(y_nc, Float64(N))
        end
    end
    if !isempty(x_nc)
        scatter!(p1, x_nc, y_nc; marker=:x, markersize=4, color=:red, label="non-converged")
    end

    omega_best = fill(NaN, nN)
    for iN in 1:nN
        valid = comp_eff[iN, :] .& conv_eff[iN, :]
        if any(valid)
            vals = zvals[iN, :]
            idx_local = argmin(vals[valid])
            idx_global = findall(valid)[idx_local]
            omega_best[iN] = omegas[idx_global]
        end
    end

    omega_theory = [2.0 / (1.0 + sin(pi / (N - 1))) for N in Ns]
    p2 = plot(
        Ns,
        omega_theory;
        label="theory",
        linestyle=:dash,
        color=:black,
        xlabel="N",
        ylabel="\\omega*",
        title="Best omega(N)",
        left_margin=12Plots.mm,
        right_margin=8Plots.mm,
        bottom_margin=8Plots.mm,
        top_margin=6Plots.mm,
        titlefontsize=10,
        guidefontsize=10,
        tickfontsize=8,
        dpi=300,
        legend=:bottomright,
    )

    finite_best = isfinite.(omega_best)
    if any(finite_best)
        scatter!(
            p2,
            Ns[finite_best],
            omega_best[finite_best];
            label="best from sweep",
            color=:blue,
            marker=:circle,
            markersize=4,
        )
    end

    p = plot(
        p1,
        p2;
        layout=grid(1, 2, widths=[0.60, 0.40]),
        size=(1900, 750),
        margin=4Plots.mm,
    )

    mkpath(dirname(output))
    savefig(p, output)
end

end
