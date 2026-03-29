# =============================================================================
# Shared helpers for Kármán vortex street solvers (LBM / FDM / FEM)
# Physical domain: Figure 4 geometry (SI units)
# =============================================================================

using Statistics, FFTW, Printf, Plots

# ── Domain constants ──────────────────────────────────────────────────────────
const KH_L  = 2.2    # domain length  (m)
const KH_H  = 0.31   # domain height  (m)  [0.16 above + 0.15 below cyl centre]
const KH_D  = 0.1    # cylinder diameter (m)
const KH_R  = 0.05   # cylinder radius   (m)
const KH_CX = 0.15   # cylinder centre x (m)
const KH_CY = 0.15   # cylinder centre y (m)

# ── Strouhal number ───────────────────────────────────────────────────────────
"""
    compute_strouhal(cl_series, dt_phys) → St

Estimate the Strouhal number from the lift-coefficient time series.
`dt_phys` is the physical time interval between consecutive samples (seconds).
Returns St = f_peak * D / U_ref  (dimensionless), using KH_D and U_ref = 1 m/s
scaled by the caller's actual U_inlet.
"""
function compute_strouhal(cl_series::Vector{Float64}, dt_phys::Float64,
                           U_inlet::Float64=1.0)
    n = length(cl_series)
    n < 16 && return 0.0
    freqs = FFTW.fftfreq(n, 1.0 / dt_phys)          # Hz
    power = abs2.(fft(cl_series .- mean(cl_series)))
    pos   = 2:n÷2
    peak  = pos[argmax(power[pos])]
    f_pk  = freqs[peak]
    return f_pk * KH_D / U_inlet
end

# ── Comparison table ──────────────────────────────────────────────────────────
"""
Result from one solver run.
Fields: solver, Re, St, Cd_mean, Cl_amp, us_per_step, max_re
"""
struct KarmanResult
    solver     :: String
    Re         :: Float64
    St         :: Float64
    Cd_mean    :: Float64
    Cl_amp     :: Float64   # half peak-to-peak
    us_per_step:: Float64   # wall-clock µs per timestep
    max_re     :: Float64   # last stable Re (=Re if no sweep)
end

function print_comparison(results::Vector{KarmanResult}, output_dir::String="plots/ass_3")
    mkpath(output_dir)
    lines = String[]
    header = @sprintf "%-6s │ Re=%5.0f │  St    │  Cd   │ Cl_amp │ µs/step │ MaxRe" "" results[1].Re
    sep    = "─"^length(header)
    push!(lines, sep)
    push!(lines, @sprintf("%-6s │ %7s │  %-6s │  %-5s │ %-6s │ %-7s │ %-6s",
          "Solver", "Re", "St", "Cd", "Cl_amp", "µs/step", "MaxRe"))
    push!(lines, sep)
    for r in results
        push!(lines, @sprintf("%-6s │ %7.0f │ %6.4f │ %5.3f │ %6.4f │ %7.1f │ %6.0f",
              r.solver, r.Re, r.St, r.Cd_mean, r.Cl_amp, r.us_per_step, r.max_re))
    end
    push!(lines, sep)
    txt = join(lines, "\n")
    println("\n" * txt * "\n")
    open(joinpath(output_dir, "comparison.txt"), "w") do io
        println(io, txt)
    end
end

# ── Velocity plot helper ──────────────────────────────────────────────────────
"""
    save_velocity_plot(ux, uy, mask, Re, solver, output_dir)

Save a velocity-magnitude heatmap.
`mask` is a Matrix{Bool} (true = obstacle) or `nothing`.
Grid axes are inferred from array dimensions using KH_L / KH_H.
"""
function save_velocity_plot(ux::Matrix{Float64}, uy::Matrix{Float64},
                             mask, Re::Float64, solver::String, output_dir::String)
    mkpath(output_dir)
    speed = sqrt.(ux .^ 2 .+ uy .^ 2)
    if mask !== nothing
        speed[mask] .= NaN
    end
    U_ref = maximum(filter(isfinite, vec(speed)))
    p = heatmap(speed';
        color=:jet, clims=(0, U_ref),
        title="$solver  Re=$(round(Int,Re))  |u|",
        xlabel="x", ylabel="y", dpi=150)
    fname = joinpath(output_dir, "$(solver)_max_Re$(round(Int,Re)).png")
    savefig(p, fname)
    println("  Saved: $fname")
end
