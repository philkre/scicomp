using Distributed
using Plots
using LaTeXStrings
using Statistics

function _resolve_output_path(output::String)::String
    if isabspath(output)
        return output
    end
    # Make relative paths deterministic: relative to laszlo/ (parent of helpers/)
    return normpath(joinpath(@__DIR__, "..", output))
end

function get_wave_plots(t_f::Float64, t_0::Float64, dt::Float64, L::Float64, N::Int, solution_1::Matrix, solution_2::Matrix, solution_3::Matrix)::Vector{Plots.Plot{Plots.GRBackend}}
    x = range(0, L, length=N)
    i_total = Int((t_f - t_0) / dt)
    plots = @distributed (vcat) for i in 1:i_total
        plot(x, solution_1[:, i], ylim=(-1, 1), title="Time: $(round(i * dt, digits=2)) s", xlabel="Position along string", ylabel="Displacement", dpi=300)
        plot!(x, solution_2[:, i])
        plot!(x, solution_3[:, i])
    end
    return plots
end

function plot_wave_multi(psis::AbstractMatrix, x::AbstractVector, title_text::String, ts_idx::AbstractVector{<:Integer}; output::String="plots/ex_1_wave_multi.png")
    p = plot(dpi=300, size=(400, 400))
    for t in ts_idx
        if t == 1
            t_label = "t=0.0"
        else
            t_label = "t=$(round((t - 1) * 0.001, digits=3))"
        end
        plot!(p, x, psis[:, t], label=t_label)
    end

    xlabel!(p, "x")
    ylabel!(p, "Psi")
    title!(p, title_text)
    output_path = _resolve_output_path(output)
    mkpath(dirname(output_path))
    savefig(p, output_path)
    return p
end

function plot_euler_leapfrog_energy(tvals::AbstractVector, energy_euler::AbstractVector, energy_leapfrog::AbstractVector; output::String="plots/ex_1_energy_euler_vs_leapfrog.png", size::Tuple{Int,Int}=(400, 400))
    p_energy = plot(
        tvals,
        energy_euler,
        label="Euler",
        xlabel="t",
        ylabel=L"$\delta E(t) = E(t) - E(0)$",
        dpi=300,
        size=size,
        color=:blue
    )
    plot!(p_energy, tvals, fill(mean(energy_euler), length(tvals)), linestyle=:dash, color=:blue, label="Euler mean")
    plot!(p_energy, tvals, energy_leapfrog, label="Leapfrog", color=:red)
    output_path = _resolve_output_path(output)
    mkpath(dirname(output_path))
    savefig(p_energy, output_path)
    return p_energy
end
