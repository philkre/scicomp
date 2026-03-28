module Assignment_3_1

using BenchmarkTools

# Import local module (general helpers)
# using Helpers


DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_3"

"""
Lattice Boltzmann Method (LBM) — Karman Vortex Street Solver
=============================================================
A minimal, educational 2D LBM solver using the D2Q9 lattice and BGK collision
operator.  Simulates flow past a circular cylinder at Reynolds number 150 to
produce the classic Karman vortex street.

Algorithm overview (each timestep):
  1. Compute macroscopic quantities (density, velocity) from distributions
  2. Collision step  — relax f toward local equilibrium (BGK)
  3. Bounce-back    — reflect populations at obstacle nodes
  4. Streaming step — propagate f_i along lattice velocity c_i
  5. Boundary conditions — Zou-He inlet, open outlet
"""
module LBM_karman

using Plots
using Statistics
using Printf

# =============================================================================
# 3.  Equilibrium Distribution Function
# =============================================================================

"""
    equilibrium(rho, ux, uy) -> Array{Float64,3}

Compute the D2Q9 equilibrium distributions for the BGK collision operator.

The equilibrium is derived from a second-order Taylor expansion of the
Maxwell-Boltzmann distribution:

    f_i^eq = w_i * rho * (1 + c_i·u/cs² + (c_i·u)²/(2·cs⁴) − u·u/(2·cs²))

where cs² = 1/3 (lattice speed of sound squared).

# Arguments
- `rho`: macroscopic density,       shape `(Nx, Ny)`
- `ux`:  x-component of velocity,   shape `(Nx, Ny)`
- `uy`:  y-component of velocity,   shape `(Nx, Ny)`

# Returns
- `feq`: equilibrium distributions, shape `(Nx, Ny, 9)`
"""
function equilibrium(rho::Matrix{Float64}, ux::Matrix{Float64}, uy::Matrix{Float64})
    feq = zeros(Nx, Ny, 9)
    usqr = @. ux^2 + uy^2                        # |u|²
    for i in 1:9
        cu = @. c[i][1] * ux + c[i][2] * uy     # c_i · u
        @. feq[:, :, i] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu^2 - 1.5 * usqr)
    end
    return feq
end

# -----------------------------------------------------------------
# 5b. Visualisation helpers
# -----------------------------------------------------------------

function plot_velocity(ux, uy, step)
    """Plot the velocity magnitude field |u|."""
    speed = sqrt.(ux .^ 2 .+ uy .^ 2)
    speed[obstacle] .= NaN   # mask cylinder
    heatmap(speed'; color=:jet, clims=(0, U_inlet * 2),
        title="Velocity magnitude — step $step", xlabel="x", ylabel="y")
    display(current())
end

function plot_vorticity(ux, uy, step)
    """
    Plot the vorticity field ω = ∂uy/∂x − ∂ux/∂y via central differences.
    Highlights the alternating vortices of the Karman street vividly.
    """
    vort = (circshift(uy, (-1, 0)) .- circshift(uy, (1, 0))
            .-
            circshift(ux, (0, -1)) .+ circshift(ux, (0, 1)))
    vort[obstacle] .= NaN
    heatmap(vort'; color=:RdBu, clims=(-0.04, 0.04),
        title="Vorticity — step $step", xlabel="x", ylabel="y")
    display(current())
end

function plot_field(ux, uy, step)
    """Dispatch to the selected visualisation mode."""
    plot_mode == "vorticity" && return plot_vorticity(ux, uy, step)
    plot_mode == "velocity" && return plot_velocity(ux, uy, step)
end

# =============================================================================
# 5.  Main Simulation
# =============================================================================

"""
    main(; n_steps, plot_every, plot_mode)

Run the LBM simulation: initialisation, time loop, and live visualisation.

# Keyword Arguments
- `n_steps`:    total timesteps (default 30 000)
- `plot_every`: plot interval in steps (default 25)
- `plot_mode`:  `"velocity"` | `"vorticity"` | `"none"` (default `"velocity"`)
"""
function main(; n_steps::Int=30_000, plot_every::Int=25, plot_mode::String="velocity")

    # =============================================================================
    # 1.  D2Q9 Lattice Definition
    # =============================================================================
    #
    #   7  3  6        Lattice velocities c_i (i = 1..9):
    #    \ | /           1: rest          (0, 0)
    #   4--1--2          2-5: axis-aligned  (±1,0), (0,±1)
    #    / | \           6-9: diagonals     (±1,±1)
    #   8  5  9
    #
    #  Each element of `c` is a (cx, cy) tuple for direction i.

    c = Tuple{Int,Int}[
        (0, 0),   # 1  — rest
        (1, 0),   # 2  — east
        (0, 1),   # 3  — north
        (-1, 0),   # 4  — west
        (0, -1),   # 5  — south
        (1, 1),   # 6  — north-east
        (-1, 1),   # 7  — north-west
        (-1, -1),   # 8  — south-west
        (1, -1),   # 9  — south-east
    ]

    # Lattice weights (from the D2Q9 equilibrium derivation)
    w = Float64[4/9,                       # rest
        1/9, 1/9, 1/9, 1/9,     # axis-aligned
        1/36, 1/36, 1/36, 1/36]    # diagonals

    # Opposite direction index for each i (used in bounce-back)
    # e.g. opposite of 2 (east) is 4 (west)
    opp = Int[1, 4, 5, 2, 3, 8, 9, 6, 7]

    # =============================================================================
    # 2.  Simulation Parameters
    # =============================================================================

    Nx = 300        # domain length  (lattice units)
    Ny = 120        # domain height  (lattice units)

    # Cylinder geometry
    cx_cyl = Nx ÷ 5   # cylinder centre x  (1/5 from inlet)
    cy_cyl = Ny ÷ 2   # cylinder centre y  (centred vertically)
    r_cyl = 8         # cylinder radius

    # Flow parameters
    U_inlet = 0.12     # inlet velocity (lattice units, keep ≪ 1 for low Mach)
    Re = 150      # target Reynolds number

    # Derived quantities:
    #   Re = U * D / nu   →  nu = U * D / Re
    #   In LBM:  nu = cs² * (tau - 0.5)  where cs² = 1/3
    #   Therefore:  tau = 3 * nu + 0.5
    D = 2 * r_cyl                    # cylinder diameter
    nu = U_inlet * D / Re             # kinematic viscosity
    tau = 3.0 * nu + 0.5              # BGK relaxation time

    println("Simulation parameters:")
    println("  Grid:      $(Nx) × $(Ny)")
    println("  Cylinder:  centre=($(cx_cyl), $(cy_cyl)),  r=$(r_cyl),  D=$(D)")
    println("  Re=$(Re),  U_inlet=$(U_inlet),  ν=$(round(nu; digits=6)),  τ=$(round(tau; digits=4))")

    # =============================================================================
    # 4.  Obstacle Mask  (circular cylinder)
    # =============================================================================

    # Coordinate grids (0-based to match physical centring)
    _X = Float64[x for x in 0:Nx-1, _ in 0:Ny-1]   # shape (Nx, Ny)
    _Y = Float64[y for _ in 0:Nx-1, y in 0:Ny-1]   # shape (Nx, Ny)

    # Boolean matrix: true where the obstacle is located
    obstacle = @. (_X - cx_cyl)^2 + (_Y - cy_cyl)^2 <= r_cyl^2

    # -----------------------------------------------------------------
    # 5a. Initialisation
    # -----------------------------------------------------------------

    # Uniform flow at inlet velocity everywhere
    rho = ones(Nx, Ny)
    ux = fill(U_inlet, Nx, Ny)
    uy = zeros(Nx, Ny)

    # Small transverse perturbation to break symmetry and trigger vortex shedding
    @. uy += 0.001 * U_inlet * sin(2π * _Y / Ny)

    # Zero velocity inside the obstacle
    ux[obstacle] .= 0.0
    uy[obstacle] .= 0.0

    # Initialise distributions to equilibrium
    f = equilibrium(rho, ux, uy)
    f_out = similar(f)

    # Precomputed (1,1,9) arrays for fast macroscopic summation
    cx_b = reshape(Float64[c[i][1] for i in 1:9], 1, 1, 9)
    cy_b = reshape(Float64[c[i][2] for i in 1:9], 1, 1, 9)



    # -----------------------------------------------------------------
    # 5c. Main time loop
    # -----------------------------------------------------------------
    #
    #  LBM "collide-then-stream" pattern each timestep:
    #    1. Macroscopic quantities (rho, u) from current distributions
    #    2. Collision:   relax f toward equilibrium  →  f_out
    #    3. Bounce-back: reflect f at obstacle nodes
    #    4. Streaming:   propagate f_out along lattice velocities  →  f
    #    5. Boundary conditions (Zou-He inlet, open outlet)
    #
    #  circshift provides periodic wrapping (y-direction boundary condition).

    println("\nRunning $n_steps timesteps …")

    for step in 1:n_steps

        # -------------------------------------------------------------
        # 7a.  Macroscopic quantities
        #      rho = Σ_i f_i,   rho·u_α = Σ_i c_{i,α} · f_i
        # -------------------------------------------------------------
        rho = dropdims(sum(f, dims=3), dims=3)
        ux = dropdims(sum(f .* cx_b, dims=3), dims=3) ./ rho
        uy = dropdims(sum(f .* cy_b, dims=3), dims=3) ./ rho

        # -------------------------------------------------------------
        # 7b.  Collision step (BGK single-relaxation-time)
        #      f_out_i = f_i − (f_i − f_i^eq) / tau
        # -------------------------------------------------------------
        feq = equilibrium(rho, ux, uy)
        @. f_out = f - (f - feq) / tau

        # -------------------------------------------------------------
        # 7c.  Bounce-back on obstacle (no-slip wall condition)
        #      Replace post-collision populations at obstacle nodes with
        #      the pre-collision populations from the opposite direction.
        # -------------------------------------------------------------
        for i in 1:9
            sl_out = view(f_out, :, :, i)
            sl_in = view(f, :, :, opp[i])
            sl_out[obstacle] .= sl_in[obstacle]
        end

        # -------------------------------------------------------------
        # 7d.  Streaming step
        #      Shift each f_i by its lattice velocity c_i.
        #      circshift gives periodic wrapping in y.
        # -------------------------------------------------------------
        for i in 1:9
            f[:, :, i] .= circshift(view(f_out, :, :, i), (c[i][1], c[i][2]))
        end

        # -------------------------------------------------------------
        # 7e.  Outlet boundary condition (zero-gradient / open)
        #      Copy from penultimate column so vortices can leave freely.
        # -------------------------------------------------------------
        f[end, :, :] .= f[end-1, :, :]

        # -------------------------------------------------------------
        # 7f.  Inlet boundary condition (Zou-He, fixed velocity)
        #      After streaming, populations east(2), ne(6), se(9) at x=1
        #      come from outside the domain and must be reconstructed.
        #      Known: rest(1), north(3), south(5), west(4), nw(7), sw(8).
        # -------------------------------------------------------------
        rho_in = (f[1, :, 1] .+ f[1, :, 3] .+ f[1, :, 5]
                  .+
                  2 .* (f[1, :, 4] .+ f[1, :, 7] .+ f[1, :, 8])
        ) ./ (1 - U_inlet)

        f[1, :, 2] .= f[1, :, 4] .+ (2 / 3) .* rho_in .* U_inlet
        f[1, :, 6] .= f[1, :, 8] .- 0.5 .* (f[1, :, 3] .- f[1, :, 5]) .+ (1 / 6) .* rho_in .* U_inlet
        f[1, :, 9] .= f[1, :, 7] .+ 0.5 .* (f[1, :, 3] .- f[1, :, 5]) .+ (1 / 6) .* rho_in .* U_inlet

        # -------------------------------------------------------------
        # 7g.  Visualisation & progress report
        # -------------------------------------------------------------
        step % plot_every == 0 && plot_field(ux, uy, step)

        if step % 1000 == 0
            avg_rho = mean(rho[.!obstacle])
            @printf "  Step %6d/%d  |  avg density = %.6f\n" step n_steps avg_rho
        end
    end

    println("\nSimulation complete.")
end

end # module


function main(;
    do_bench::Bool=false,
    do_cache::Bool=false,
    plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR,
    do_gif::Bool=false,
)
    if do_bench
        @time "Done benchmarking" begin
            # TODO
        end
    end

    # TODO
    if do_gif
        println("OBAMA GIF MADE")
    end

    return
end

end # module
