import Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()

include("model.jl")
include("data.jl")
include("sim.jl")
include("plot.jl")

using .Sim: run_dla
using .Plotting: animate_dla, plot_steadystate
using Plots: palette, plot, plot!, savefig
using Measures: mm
using Polynomials: coeffs, fit
using LaTeXStrings
using Statistics: mean, std
using Measures: mm

"""
    cluster_stats(m; occupied_value=0.0, normalized=false)

Compute cluster statistics from occupancy map `m` (`occupied_value` marks particles).
Returns `(n_particles, r_g)` where `r_g` is the radius of gyration about the
cluster center of mass.

If `normalized=true`, coordinates are mapped to `[0,1] x [0,1]`; otherwise
grid indices are used.
"""
function cluster_stats(
    m::AbstractMatrix{<:Real};
    occupied_value::Float64=0.0,
    normalized::Bool=false,
)
    occ = findall(x -> x == occupied_value, m)
    n_particles = length(occ)
    if n_particles == 0
        return 0, 0.0
    end

    ni, nj = size(m)
    xs = Vector{Float64}(undef, n_particles)
    ys = Vector{Float64}(undef, n_particles)

    @inbounds for (k, I) in enumerate(occ)
        i, j = Tuple(I)
        if normalized
            xs[k] = (i - 1) / max(1, ni - 1)
            ys[k] = (j - 1) / max(1, nj - 1)
        else
            xs[k] = i
            ys[k] = j
        end
    end

    x_cm = sum(xs) / n_particles
    y_cm = sum(ys) / n_particles
    r2 = 0.0
    @inbounds @simd for k in 1:n_particles
        dx = xs[k] - x_cm
        dy = ys[k] - y_cm
        r2 += dx * dx + dy * dy
    end
    r_g = sqrt(r2 / n_particles)

    return n_particles, r_g
end

"""
    main_dla()

Run the default DLA experiment, print timing breakdown, and save the final
steady-state concentration plot to disk.
"""
function main_dla()
    N = 100
    max_cycles = 20
    eta = 1.5
    omega = 1.6
    epsilon = 1e-6
    growth_steps = 500

    PLAYGROUND = false
    ETA_SWEEP = true

    if PLAYGROUND

        results, to = run_dla(
            N, max_cycles, eta, omega;
            epsilon=epsilon,
            growth_steps=growth_steps,
            timed=true,
        )

        animate_dla(results; filename="philipp/output/img/dla_animation.mp4", fps=30, max_frames=300)

        println(to)

        plot_steadystate(
            results["cs"][end],
            "philipp/output/img/dla_concentration.png";
            sink_indices=findall(results["masks"][end] .== 0.0),
        )
    end

    if ETA_SWEEP
        # eta range
        eta_values = 0.1:0.1:2.0
        colors = palette(:viridis, length(eta_values))
        iterations = 30

        p1 = plot(
            xlabel=L"R_g",
            ylabel=L"M",
            title="Particle Count vs Gyration Radius",
            legend=:topleft,
            dpi=300
        )

        p2 = plot(
            xlabel=L"\eta",
            ylabel=L"D",
            title="Fractal Dimension vs eta",
            legend=false,
            dpi=300
        )

        # for each eta, compute (M, R_g) at every stored timestep and draw one curve
        for (k, eta) in enumerate(eta_values)
            Mss = Vector{Vector{Float64}}()
            Rgss = Vector{Vector{Float64}}()
            Ds = Float64[]
            for _ in 1:iterations
                results = run_dla(
                    N, max_cycles, eta, omega;
                    epsilon=epsilon,
                    growth_steps=growth_steps,
                    timed=false,
                )

                Ms = Float64[]
                Rgs = Float64[]
                for m in results["masks"]
                    n_particles, r_g = cluster_stats(m)
                    push!(Ms, n_particles)
                    push!(Rgs, r_g)
                end

                # per-run fractal dimension estimate for uncertainty bars
                mask_it = isfinite.(Rgs) .& isfinite.(Ms) .& (Rgs .> 0) .& (Ms .> 0)
                x_it = log.(Rgs[mask_it])
                y_it = log.(Ms[mask_it])
                if length(x_it) >= 2
                    pfit_it = fit(x_it, y_it, 1)
                    push!(Ds, coeffs(pfit_it)[2])
                end

                push!(Mss, Ms)
                push!(Rgss, Rgs)
            end

            # align variable-length trajectories to common prefix and aggregate
            common_len = minimum(length.(Mss))
            if common_len < 2
                @warn "Skipping eta=$eta due to too-short trajectories" common_len
                continue
            end
            Mmat = reduce(hcat, [v[1:common_len] for v in Mss])
            Rgmat = reduce(hcat, [v[1:common_len] for v in Rgss])

            Ms = vec(mean(Mmat, dims=2))
            Rgs = vec(mean(Rgmat, dims=2))
            Ms_std = vec(std(Mmat, dims=2))
            Rgs_std = vec(std(Rgmat, dims=2))

            D_mean = isempty(Ds) ? D : mean(Ds)
            D_std = length(Ds) > 1 ? std(Ds) : 0.0

            plot!(
                p1,
                Rgs,
                Ms;
                label="eta=$(round(eta; digits=2))",
                color=colors[k],
                markersize=2,
                linewidth=1.5,
                ribbon=Ms_std,
                fillalpha=0.10,
            )

            plot!(
                p2,
                [eta],
                [D_mean];
                seriestype=:scatter,
                yerror=[D_std],
                label="",
                color=colors[k],
                markersize=6,
            )
        end

        p = plot(p1, p2; layout=(1, 2), size=(1200, 500), dpi=300, left_margin=8mm, right_margin=8mm, top_margin=4mm, bottom_margin=10mm)
        outfile = "philipp/output/img/eta_sweep_M_vs_Rg.png"
        mkpath(dirname(outfile))
        savefig(p, outfile)
    end
end

main_dla()
