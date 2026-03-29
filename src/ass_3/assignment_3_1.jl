module Assignment_3_1

using BenchmarkTools

include("../helpers/karman_helpers.jl")

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

# =============================================================================
# 5.  Main Simulation
# =============================================================================

"""
    main(; n_steps, plot_every, plot_mode, Re, save_path, do_gif, gif_path)

Run the LBM simulation: initialisation, time loop, and live visualisation.

# Keyword Arguments
- `n_steps`:    total timesteps (default 30 000)
- `plot_every`: plot interval in steps (default 25)
- `plot_mode`:  `"velocity"` | `"vorticity"` | `"none"` (default `"velocity"`)
- `Re`:         Reynolds number (default 150)
- `save_path`:  path to save final frame PNG (default "" = don't save)
- `do_gif`:     collect frames for animation (default false)
- `gif_path`:   path to save GIF (default "" = don't save)
"""
function main(; n_steps::Int=30_000, plot_every::Int=25, plot_mode::String="velocity",
    Re::Float64=150.0, save_path::String="", do_gif::Bool=false, gif_path::String="")

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

    # Physical dimensions (SI): W=2.2m, H=0.31m, D_cyl=0.1m, centre at (0.15m, 0.15m)
    # Resolution: D_cyl mapped to 32 lattice nodes → Δx = 0.1/32 m
    # (D=32 keeps tau > 0.55 up to Re≈350, giving useful stability headroom)
    r_cyl = 16         # cylinder radius  (lattice units)
    D_phys = 0.1        # cylinder diameter (m)
    dx = D_phys / (2 * r_cyl)   # lattice spacing (m/node)

    Nx = round(Int, 2.2 / dx)   # 704  — domain length
    Ny = round(Int, 0.31 / dx)   # 99   — domain height  (0.16 + 0.15 m)

    # Cylinder centre: 0.15 m from inlet, 0.15 m from bottom wall
    cx_cyl = round(Int, 0.15 / dx)   # 48
    cy_cyl = round(Int, 0.15 / dx)   # 48  (slightly below mid → triggers shedding)

    # Flow parameters
    U_inlet = 0.12     # inlet velocity (lattice units, keep ≪ 1 for low Mach)

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

    # Equilibrium distribution (closure — captures Nx, Ny, c, w)
    function equilibrium(rho::Matrix{Float64}, ux::Matrix{Float64}, uy::Matrix{Float64})
        feq = zeros(Nx, Ny, 9)
        usqr = @. ux^2 + uy^2
        for i in 1:9
            cu = @. c[i][1] * ux + c[i][2] * uy
            @. feq[:, :, i] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu^2 - 1.5 * usqr)
        end
        return feq
    end

    # Initialise distributions to equilibrium
    f = equilibrium(rho, ux, uy)
    f_out = similar(f)

    # Precomputed (1,1,9) arrays for fast macroscopic summation
    cx_b = reshape(Float64[c[i][1] for i in 1:9], 1, 1, 9)
    cy_b = reshape(Float64[c[i][2] for i in 1:9], 1, 1, 9)

    # -----------------------------------------------------------------
    # 5b. Visualisation helpers (closures capturing obstacle, U_inlet)
    # -----------------------------------------------------------------

    function plot_velocity_field(ux, uy, step)
        speed = sqrt.(ux .^ 2 .+ uy .^ 2)
        speed[obstacle] .= NaN
        heatmap(speed'; color=:jet, clims=(0, U_inlet * 2),
            title="Re=$(round(Int,Re))  |u|  step $step", xlabel="x", ylabel="y",
            aspect_ratio=:equal, dpi=300)
    end

    function plot_vorticity_field(ux, uy, step)
        vort = (circshift(uy, (-1, 0)) .- circshift(uy, (1, 0))
                .-
                circshift(ux, (0, -1)) .+ circshift(ux, (0, 1)))
        vort[obstacle] .= NaN
        heatmap(vort'; color=:RdBu, clims=(-0.04, 0.04),
            title="Re=$(round(Int,Re))  ω  step $step", xlabel="x", ylabel="y",
            aspect_ratio=:equal, dpi=300)
    end

    function plot_field(ux, uy, step)
        plot_mode == "vorticity" && return plot_vorticity_field(ux, uy, step)
        plot_mode == "velocity" && return plot_velocity_field(ux, uy, step)
    end

    anim = do_gif ? Animation() : nothing

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
        #      x: open (handled by inlet/outlet BCs below)
        #      y: no-slip walls — bounce-back replaces periodic wrap
        # -------------------------------------------------------------
        for i in 1:9
            cx_i, cy_i = c[i]
            # Manual shift with wall bounce-back in y
            for jj in 1:Ny
                jj_src = jj - cy_i
                if jj_src < 1 || jj_src > Ny
                    # Node would come from outside domain → bounce back
                    f[:, jj, i] .= view(f_out, :, jj, opp[i])
                else
                    for ii in 1:Nx
                        ii_src = ii - cx_i
                        if 1 <= ii_src <= Nx
                            f[ii, jj, i] = f_out[ii_src, jj_src, i]
                        end
                    end
                end
            end
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
        if step % plot_every == 0 && plot_mode != "none"
            plot_field(ux, uy, step)
            do_gif && frame(anim)
        end

        if step % 1000 == 0
            avg_rho = mean(rho[.!obstacle])
            @printf "  Step %6d/%d  |  avg density = %.6f\n" step n_steps avg_rho
        end
    end

    println("\nSimulation complete.")

    # Save final frame
    if !isempty(save_path) && plot_mode != "none"
        plot_velocity_field(ux, uy, n_steps)
        mkpath(dirname(save_path))
        savefig(save_path)
        println("  Saved: $save_path")
    end

    # Save GIF
    if do_gif && !isempty(gif_path)
        mkpath(dirname(gif_path))
        gif(anim, gif_path; fps=20)
        println("  Saved GIF: $gif_path")
    end
end

end # module LBM_karman


# =============================================================================
# FDM Kármán Vortex Street Solver — MAC staggered grid, fractional step
# =============================================================================
module FDM_karman

using Statistics, Printf, LinearAlgebra

# Returns true when both Cl amplitude and mean Cd have stabilised over the
# last 2*window recorded samples (relative change < tol in each half).
function _converged(cd::Vector{Float64}, cl::Vector{Float64},
    window::Int, tol::Float64)
    length(cl) < 2 * window && return false
    a_cl = cl[end-2*window+1:end-window]
    b_cl = cl[end-window+1:end]
    amp_a = maximum(a_cl) - minimum(a_cl)
    amp_b = maximum(b_cl) - minimum(b_cl)
    amp_ref = max(amp_a, amp_b, 1e-10)
    amp_ok = abs(amp_a - amp_b) / amp_ref < tol

    a_cd = cd[end-2*window+1:end-window]
    b_cd = cd[end-window+1:end]
    mean_a = mean(a_cd)
    mean_b = mean(b_cd)
    mean_ref = max(abs(mean_a), abs(mean_b), 1e-10)
    mean_ok = abs(mean_a - mean_b) / mean_ref < tol

    return amp_ok && mean_ok
end

"""
    main(; Re, Nx, Ny, max_steps, conv_tol, conv_window, record_every,
           plot_output_dir, sweep_re)

Run the FDM fractional-step Navier-Stokes simulation on a MAC staggered grid.
Runs until the Cl amplitude and mean Cd stabilise (convergence) or `max_steps`
is reached. Returns a NamedTuple with cd, cl, steps, elapsed_ns, max_re.
"""
function main(;
    Re::Float64=100.0,
    Nx::Int=220,
    Ny::Int=62,
    max_steps::Int=12_000,
    conv_tol::Float64=0.02,
    conv_window::Int=60,
    record_every::Int=20,
    plot_output_dir::String="plots/ass_3",
    sweep_re::Bool=false,
)::NamedTuple

    # ── Physical constants (from karman_helpers.jl scope) ─────────────────────
    L = Main.Assignment_3_1.KH_L    # 2.2 m
    H = Main.Assignment_3_1.KH_H    # 0.31 m
    D = Main.Assignment_3_1.KH_D    # 0.1 m
    R = Main.Assignment_3_1.KH_R    # 0.05 m
    CX = Main.Assignment_3_1.KH_CX   # 0.15 m
    CY = Main.Assignment_3_1.KH_CY   # 0.15 m

    # ── Re sweep logic ─────────────────────────────────────────────────────────
    re_list = sweep_re ? [100.0, 200.0, 300.0, 500.0, 1000.0] : [Re]

    all_cd = Float64[]
    all_cl = Float64[]
    all_steps = Float64[]
    last_stable_Re = re_list[1]
    elapsed_total = UInt64(0)

    for cur_Re in re_list
        t0 = time_ns()

        # ── Grid spacing ───────────────────────────────────────────────────────
        dx = L / Nx
        dy = H / Ny

        # ── Flow parameters ────────────────────────────────────────────────────
        U_inlet = 0.1                   # m/s (physical, Re=100 → ν=U*D/Re=1e-4)
        nu = U_inlet * D / cur_Re  # kinematic viscosity

        # ── Adaptive time step (CFL + diffusion limits) ────────────────────────
        dt = 0.25 * min(dx, dy) / max(U_inlet, 1e-6)
        dt = min(dt, 0.1 * min(dx, dy)^2 / nu)

        @printf("FDM Re=%.0f  Nx=%d Ny=%d  dx=%.4f dy=%.4f  dt=%.6f  nu=%.6f\n",
            cur_Re, Nx, Ny, dx, dy, dt, nu)

        # ── Allocate velocity/pressure arrays ─────────────────────────────────
        # u: x-velocity at (i-½, j), size (Nx+1) × Ny, i=1..Nx+1, j=1..Ny
        # v: y-velocity at (i, j-½), size  Nx × (Ny+1), i=1..Nx,  j=1..Ny+1
        # p: pressure at cell centre, size  Nx × Ny
        u = zeros(Nx + 1, Ny)
        v = zeros(Nx, Ny + 1)
        p = zeros(Nx, Ny)
        u_star = zeros(Nx + 1, Ny)
        v_star = zeros(Nx, Ny + 1)
        rhs = zeros(Nx, Ny)

        # Initialise with uniform inlet flow
        fill!(u, U_inlet)

        # ── Immersed boundary masks ────────────────────────────────────────────
        obs_p = falses(Nx, Ny)     # cell centres inside cylinder
        obs_u = falses(Nx + 1, Ny)     # u-faces inside cylinder
        obs_v = falses(Nx, Ny + 1)   # v-faces inside cylinder

        for j in 1:Ny, i in 1:Nx
            xc = (i - 0.5) * dx
            yc = (j - 0.5) * dy
            obs_p[i, j] = (xc - CX)^2 + (yc - CY)^2 <= R^2
        end
        for j in 1:Ny, i in 1:Nx+1
            xu = (i - 1) * dx
            yu = (j - 0.5) * dy
            obs_u[i, j] = (xu - CX)^2 + (yu - CY)^2 <= R^2
        end
        for j in 1:Ny+1, i in 1:Nx
            xv = (i - 0.5) * dx
            yv = (j - 1) * dy
            obs_v[i, j] = (xv - CX)^2 + (yv - CY)^2 <= R^2
        end

        # Zero initial velocity inside obstacle
        u[obs_u] .= 0.0
        v[obs_v] .= 0.0

        # ── Time series ────────────────────────────────────────────────────────
        cd_series = Float64[]
        cl_series = Float64[]
        step_times = Float64[]
        diverged = false

        # ── SOR parameters ────────────────────────────────────────────────────
        omega = 1.85  # near-optimal for 220×62
        sor_iter = 400
        sor_tol = 1e-4

        # ── Precompute reciprocals ─────────────────────────────────────────────
        idx = 1.0 / dx
        idy = 1.0 / dy
        idx2 = idx^2
        idy2 = idy^2
        coeff_p = dx^2 * dy^2 / (2.0 * (dx^2 + dy^2))

        # ── Main time loop ─────────────────────────────────────────────────────
        step = 0
        while step < max_steps
            step += 1
            ts = time_ns()

            # ------------------------------------------------------------------
            # Stage 1: Intermediate velocity u*, v*
            # ------------------------------------------------------------------
            @views begin
                # --- u* (interior u-faces: i=2..Nx, j=1..Ny) ---
                for j in 1:Ny
                    for i in 2:Nx
                        # Laplacian
                        # ghost cells for top/bottom walls (no-slip): u[i,0]=-u[i,1], u[i,Ny+1]=-u[i,Ny]
                        u_ij = u[i, j]
                        u_ip1j = u[i+1, j]
                        u_im1j = u[i-1, j]
                        u_ijp1 = j < Ny ? u[i, j+1] : -u[i, j]
                        u_ijm1 = j > 1 ? u[i, j-1] : -u[i, j]

                        lap_u = (u_ip1j - 2 * u_ij + u_im1j) * idx2 +
                                (u_ijp1 - 2 * u_ij + u_ijm1) * idy2

                        # Advection (central differences)
                        # d(u²)/dx: u-bar at i+½ and i-½
                        u_e = 0.5 * (u[i, j] + u[i+1, j])
                        u_w = 0.5 * (u[i-1, j] + u[i, j])
                        adv_uu = (u_e^2 - u_w^2) * idx

                        # d(uv)/dy: v interpolated to u-face
                        # v at (i, j+½) interpolated to u-face (i-½, j+½):
                        # use the 4 surrounding v nodes
                        v_n_l = j <= Ny ? v[i-1, j+1] : 0.0
                        v_n_r = j <= Ny ? v[i, j+1] : 0.0
                        v_s_l = v[i-1, j]
                        v_s_r = v[i, j]
                        # safe index: i-1 >= 1 since i starts at 2
                        v_north = 0.25 * (v_n_l + v_n_r + v_s_l + v_s_r)
                        # v at (i, j-½) interpolated to u-face
                        v_ss_l = j > 1 ? v[i-1, j] : 0.0
                        v_ss_r = j > 1 ? v[i, j] : 0.0
                        v_sss_l = j > 1 ? v[i-1, j-1] : 0.0
                        v_sss_r = j > 1 ? v[i, j-1] : 0.0
                        v_south = 0.25 * (v_ss_l + v_ss_r + v_sss_l + v_sss_r)

                        u_mid = u[i, j]
                        adv_uv = (u_mid * v_north - u_mid * v_south) * idy

                        u_star[i, j] = u_ij + dt * (nu * lap_u - adv_uu - adv_uv)
                    end
                end

                # --- v* (interior v-faces: i=1..Nx, j=2..Ny) ---
                for j in 2:Ny
                    for i in 1:Nx
                        v_ij = v[i, j]
                        v_ip1j = i < Nx ? v[i+1, j] : v[i, j]  # zero-gradient at outlet
                        v_im1j = i > 1 ? v[i-1, j] : 0.0      # inlet: v=0
                        v_ijp1 = j < Ny ? v[i, j+1] : -v[i, j]  # no-slip top ghost (actually wall BC overrides)
                        v_ijm1 = v[i, j-1]

                        lap_v = (v_ip1j - 2 * v_ij + v_im1j) * idx2 +
                                (v_ijp1 - 2 * v_ij + v_ijm1) * idy2

                        # d(v²)/dy
                        v_n = 0.5 * (v[i, j] + (j < Ny ? v[i, j+1] : v[i, j]))
                        v_s = 0.5 * (v[i, j-1] + v[i, j])
                        adv_vv = (v_n^2 - v_s^2) * idy

                        # d(uv)/dx: u interpolated to v-face
                        u_e_t = 0.25 * (u[i+1, j-1] + u[i+1, j] + u[i, j-1] + u[i, j])
                        u_w_t = i > 1 ? 0.25 * (u[i, j-1] + u[i, j] + u[i-1, j-1] + u[i-1, j]) :
                                0.25 * (u[1, j-1] + u[1, j] + 0.0 + 0.0)

                        v_mid = v[i, j]
                        adv_uv = (u_e_t * v_mid - u_w_t * v_mid) * idx

                        v_star[i, j] = v_ij + dt * (nu * lap_v - adv_uv - adv_vv)
                    end
                end
            end # @views

            # -- Apply IB (zero inside cylinder) and BCs after stage 1 --
            u_star[obs_u] .= 0.0
            v_star[obs_v] .= 0.0

            # u* BCs: inlet and outlet
            for j in 1:Ny
                u_star[1, j] = U_inlet
                u_star[Nx+1, j] = u_star[Nx, j]   # zero-gradient outlet
            end
            # v* BCs: inlet, outlet, top, bottom
            for j in 1:Ny+1
                v_star[1, j] = 0.0
                v_star[Nx, j] = v_star[Nx-1, j]
            end
            for i in 1:Nx
                v_star[i, 1] = 0.0
                v_star[i, Ny+1] = 0.0
            end

            # ------------------------------------------------------------------
            # Stage 2: Pressure Poisson (SOR)
            # ------------------------------------------------------------------
            # Compute divergence of u*
            @views for j in 1:Ny, i in 1:Nx
                rhs[i, j] = ((u_star[i+1, j] - u_star[i, j]) * idx +
                             (v_star[i, j+1] - v_star[i, j]) * idy) / dt
            end

            # SOR iterations
            for _ in 1:sor_iter
                res = 0.0
                for j in 1:Ny
                    for i in 1:Nx
                        # Neumann BCs: ghost cell = neighbour (zero normal gradient)
                        p_e = i < Nx ? p[i+1, j] : p[i, j]    # outlet: Dirichlet p=0 handled below
                        p_w = i > 1 ? p[i-1, j] : p[i, j]
                        p_n = j < Ny ? p[i, j+1] : p[i, j]
                        p_s = j > 1 ? p[i, j-1] : p[i, j]

                        p_new_ij = (p_e + p_w) * idx2 + (p_n + p_s) * idy2
                        p_new_ij = (p_new_ij - rhs[i, j]) / (2.0 * (idx2 + idy2))

                        p_sor = (1.0 - omega) * p[i, j] + omega * p_new_ij
                        res += (p_sor - p[i, j])^2
                        p[i, j] = p_sor
                    end
                end
                # Outlet Dirichlet: p[Nx,:] = 0
                @views p[Nx, :] .= 0.0

                sqrt(res / (Nx * Ny)) < sor_tol && break
            end

            # ------------------------------------------------------------------
            # Stage 3: Velocity correction
            # ------------------------------------------------------------------
            @views for j in 1:Ny, i in 2:Nx
                u[i, j] = u_star[i, j] - dt * (p[i, j] - p[i-1, j]) * idx
            end
            @views for j in 2:Ny, i in 1:Nx
                v[i, j] = v_star[i, j] - dt * (p[i, j] - p[i, j-1]) * idy
            end

            # -- Apply IB and BCs after correction --
            u[obs_u] .= 0.0
            v[obs_v] .= 0.0

            # u BCs
            for j in 1:Ny
                u[1, j] = U_inlet
                u[Nx+1, j] = u[Nx, j]
            end
            # v BCs
            for j in 1:Ny+1
                v[1, j] = 0.0
                v[Nx, j] = v[Nx-1, j]
            end
            for i in 1:Nx
                v[i, 1] = 0.0
                v[i, Ny+1] = 0.0
            end

            # ------------------------------------------------------------------
            # Force extraction (record every `record_every` steps)
            # ------------------------------------------------------------------
            if step % record_every == 0
                Fx = 0.0
                Fy = 0.0
                for jj in 1:Ny, ii in 1:Nx
                    obs_p[ii, jj] && continue
                    if ii < Nx && obs_p[ii+1, jj]
                        Fx += p[ii, jj] * dy
                    end
                    if ii > 1 && obs_p[ii-1, jj]
                        Fx -= p[ii, jj] * dy
                    end
                    if jj < Ny && obs_p[ii, jj+1]
                        Fy += p[ii, jj] * dx
                    end
                    if jj > 1 && obs_p[ii, jj-1]
                        Fy -= p[ii, jj] * dx
                    end
                end
                Cd = 2 * Fx / (U_inlet^2 * D)
                Cl = 2 * Fy / (U_inlet^2 * D)
                push!(cd_series, Cd)
                push!(cl_series, Cl)
            end

            te = time_ns()
            push!(step_times, Float64(te - ts) * 1e-3)   # µs

            # Periodic progress print
            if step % 2000 == 0
                @printf("  FDM Re=%.0f  step=%d  Cd=%.4f\n",
                    cur_Re, step, isempty(cd_series) ? 0.0 : last(cd_series))
            end

            # Convergence check
            if _converged(cd_series, cl_series, conv_window, conv_tol)
                @printf("  FDM Re=%.0f  converged at step %d  (%.0f samples)\n",
                    cur_Re, step, length(cl_series))
                break
            end
        end # time loop

        t1 = time_ns()
        elapsed_total += t1 - t0

        # Check divergence
        if any(isnan.(cd_series)) || any(abs.(cl_series) .> 100)
            diverged = true
            @printf "  Re=%.0f diverged — stopping sweep\n" cur_Re
        else
            last_stable_Re = cur_Re
            # Save velocity plot for this Re
            # Build cell-centre u (average of adjacent u-faces)
            ux_cc = 0.5 * (u[1:Nx, :] .+ u[2:Nx+1, :])
            # Build cell-centre v (average of adjacent v-faces)
            vy_cc = 0.5 * (v[:, 1:Ny] .+ v[:, 2:Ny+1])
            Main.Assignment_3_1.save_velocity_plot(
                ux_cc, vy_cc, obs_p, cur_Re, "FDM", plot_output_dir)
            @printf("  Re=%.0f stable  Cd_mean=%.4f  Cl_amp=%.4f\n",
                cur_Re, mean(cd_series),
                isempty(cl_series) ? 0.0 : 0.5 * (maximum(cl_series) - minimum(cl_series)))
        end

        append!(all_cd, cd_series)
        append!(all_cl, cl_series)
        append!(all_steps, step_times)

        diverged && break
    end # Re sweep

    return (cd=all_cd, cl=all_cl, steps=all_steps,
        elapsed_ns=elapsed_total, max_re=last_stable_Re)
end

end # module FDM_karman


# =============================================================================
# FEM Kármán Vortex Street Solver — Gridap Taylor-Hood P2/P1, BDF2
# =============================================================================
module FEM_karman

using Statistics, Printf, LinearAlgebra, SparseArrays
using Gridap
using Gridap.Fields, Gridap.ReferenceFEs, Gridap.Geometry
using Gridap.FESpaces, Gridap.MultiField
using Gridap.Algebra
using GridapGmsh
import Gmsh

# ── Mesh generation ───────────────────────────────────────────────────────────
"""
Generate the Kármán channel mesh with a circular hole and write to `msh_path`.
Reuses the file if it already exists.
"""
function _ensure_mesh(msh_path::String, lc_cyl::Float64, lc_far::Float64)
    isfile(msh_path) && return
    mkpath(dirname(msh_path))
    L = Main.Assignment_3_1.KH_L
    H = Main.Assignment_3_1.KH_H
    R = Main.Assignment_3_1.KH_R
    CX = Main.Assignment_3_1.KH_CX
    CY = Main.Assignment_3_1.KH_CY

    Gmsh.initialize()
    Gmsh.option.setNumber("General.Terminal", 0)
    Gmsh.model.add("karman")
    factory = Gmsh.model.geo

    # Corner points of rectangle
    p1 = factory.addPoint(0.0, 0.0, 0.0, lc_far)
    p2 = factory.addPoint(L, 0.0, 0.0, lc_far)
    p3 = factory.addPoint(L, H, 0.0, lc_far)
    p4 = factory.addPoint(0.0, H, 0.0, lc_far)

    # Rectangle lines
    l1 = factory.addLine(p1, p2)
    l2 = factory.addLine(p2, p3)
    l3 = factory.addLine(p3, p4)
    l4 = factory.addLine(p4, p1)

    # Cylinder: 4 arc segments
    pc = factory.addPoint(CX, CY, 0.0, lc_cyl)
    pr = factory.addPoint(CX + R, CY, 0.0, lc_cyl)
    pt = factory.addPoint(CX, CY + R, 0.0, lc_cyl)
    pl = factory.addPoint(CX - R, CY, 0.0, lc_cyl)
    pb = factory.addPoint(CX, CY - R, 0.0, lc_cyl)
    a1 = factory.addCircleArc(pr, pc, pt)
    a2 = factory.addCircleArc(pt, pc, pl)
    a3 = factory.addCircleArc(pl, pc, pb)
    a4 = factory.addCircleArc(pb, pc, pr)

    # Curve loops and surface with hole
    cl_rect = factory.addCurveLoop([l1, l2, l3, l4])
    cl_cyl = factory.addCurveLoop([a1, a2, a3, a4])
    surf = factory.addPlaneSurface([cl_rect, cl_cyl])

    factory.synchronize()

    # Physical groups (tags for BCs)
    Gmsh.model.addPhysicalGroup(1, [l4], 1)  # inlet
    Gmsh.model.addPhysicalGroup(1, [l2], 2)  # outlet
    Gmsh.model.addPhysicalGroup(1, [l1, l3], 3)  # walls (top+bottom)
    Gmsh.model.addPhysicalGroup(1, [a1, a2, a3, a4], 4)  # cylinder
    Gmsh.model.addPhysicalGroup(2, [surf], 5)  # fluid domain

    Gmsh.model.setPhysicalName(1, 1, "inlet")
    Gmsh.model.setPhysicalName(1, 2, "outlet")
    Gmsh.model.setPhysicalName(1, 3, "walls")
    Gmsh.model.setPhysicalName(1, 4, "cylinder")
    Gmsh.model.setPhysicalName(2, 5, "fluid")

    Gmsh.model.mesh.generate(2)
    Gmsh.write(msh_path)
    Gmsh.finalize()
    println("  Mesh written to $msh_path")
end

# Shared convergence helper (same logic as FDM_karman._converged)
function _converged(cd::Vector{Float64}, cl::Vector{Float64},
    window::Int, tol::Float64)
    length(cl) < 2 * window && return false
    a_cl = cl[end-2*window+1:end-window]
    b_cl = cl[end-window+1:end]
    amp_a = maximum(a_cl) - minimum(a_cl)
    amp_b = maximum(b_cl) - minimum(b_cl)
    amp_ref = max(amp_a, amp_b, 1e-10)
    amp_ok = abs(amp_a - amp_b) / amp_ref < tol

    a_cd = cd[end-2*window+1:end-window]
    b_cd = cd[end-window+1:end]
    mean_a = mean(a_cd)
    mean_b = mean(b_cd)
    mean_ref = max(abs(mean_a), abs(mean_b), 1e-10)
    mean_ok = abs(mean_a - mean_b) / mean_ref < tol

    return amp_ok && mean_ok
end

# ── Main solver ───────────────────────────────────────────────────────────────
"""
    main(; Re, max_steps, conv_tol, conv_window, record_every, plot_output_dir, sweep_re)

Run the FEM incompressible Navier-Stokes simulation using Gridap Taylor-Hood
P2/P1 elements with BDF2 time integration. Runs until Cl amplitude and mean Cd
stabilise or `max_steps` is reached. Returns a NamedTuple with
cd, cl, steps, elapsed_ns, max_re.
"""
function main(;
    Re::Float64=100.0,
    max_steps::Int=1_500,
    conv_tol::Float64=0.05,
    conv_window::Int=15,
    record_every::Int=20,
    h_cyl::Float64=0.03,
    h_far::Float64=0.05,
    plot_output_dir::String="plots/ass_3",
    sweep_re::Bool=false,
)::NamedTuple

    D_phys = Main.Assignment_3_1.KH_D
    U_inlet = 0.1   # m/s

    re_list = sweep_re ? [100.0, 200.0, 300.0, 500.0, 1000.0] : [Re]

    all_cd = Float64[]
    all_cl = Float64[]
    all_steps = Float64[]
    last_stable_Re = re_list[1]
    elapsed_total = UInt64(0)

    msh_path = joinpath(plot_output_dir,
        "karman_mesh_hc$(round(Int,1000*h_cyl))_hf$(round(Int,1000*h_far)).msh")
    _ensure_mesh(msh_path, h_cyl, h_far)

    for cur_Re in re_list
        nu = U_inlet * D_phys / cur_Re

        @printf "FEM Re=%.0f  nu=%.6f\n" cur_Re nu

        t0 = time_ns()

        # ── Load mesh ──────────────────────────────────────────────────────────
        model = GmshDiscreteModel(msh_path)

        # ── FE spaces: Taylor-Hood P2/P1 ──────────────────────────────────────
        order = 2
        refFEu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
        refFEp = ReferenceFE(lagrangian, Float64, order - 1)

        # Dirichlet tags: inlet(1), walls(3), cylinder(4)
        u_inlet_val = VectorValue(U_inlet, 0.0)
        u_zero = VectorValue(0.0, 0.0)

        V = TestFESpace(model, refFEu; conformity=:H1,
            dirichlet_tags=["inlet", "walls", "cylinder"],
            dirichlet_masks=[(true, true), (true, true), (true, true)])
        Q = TestFESpace(model, refFEp; conformity=:H1)

        U_space = TrialFESpace(V, [u_inlet_val, u_zero, u_zero])
        P_space = TrialFESpace(Q)

        Y = MultiFieldFESpace([V, Q])
        X = MultiFieldFESpace([U_space, P_space])

        # ── Integration measures ──────────────────────────────────────────────
        degree = 2 * order
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree)

        Γ_cyl = BoundaryTriangulation(model; tags=["cylinder"])
        dΓ = Measure(Γ_cyl, degree)
        n_cyl = get_normal_vector(Γ_cyl)

        # ── Initial condition (uniform flow) ──────────────────────────────────
        u0_h = interpolate_everywhere(u_inlet_val, U_space)
        p0_h = interpolate_everywhere(0.0, P_space)

        # BDF2 needs two previous steps; prime with u0 for both
        u_nm1 = u0_h   # u^{n-1}
        u_n = u0_h   # u^n

        # ── Adaptive dt based on CFL ──────────────────────────────────────────
        dt = 0.4 * h_cyl / U_inlet

        cd_series = Float64[]
        cl_series = Float64[]
        step_times = Float64[]
        diverged = false

        step = 0
        while step < max_steps
            step += 1
            ts = time_ns()

            # BDF2 coefficients: (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = rhs
            bdf2_coeff = 1.0 / (2.0 * dt)

            # Nonlinear weak form: BDF2 + advection (semi-implicit: linearise
            # advection around u^n) + diffusion + pressure
            #   a(u,v) = ∫ [ (3/(2dt)) u·v + ν ∇u:∇v + (u^n·∇u)·v - p ∇·v + q ∇·u ] dΩ
            #   l(v)   = ∫ [ (4u^n - u^{n-1})/(2dt) · v ] dΩ

            u_adv = u_n   # linearisation point

            rhs_bdf = u_adv -> (4.0 .* u_n - u_nm1) * bdf2_coeff

            a((u, p), (v, q)) =
                ∫((3 * bdf2_coeff) * (u ⊙ v)
                  + nu * (∇(u) ⊙ ∇(v))
                  + ((∇(u)' ⋅ u_adv) ⊙ v)
                  -
                  p * (∇ ⋅ v)
                  +
                  q * (∇ ⋅ u)
                )dΩ

            l((v, q)) =
                ∫(((4.0 .* u_n - u_nm1) * bdf2_coeff) ⊙ v)dΩ

            # Assemble and solve
            op = AffineFEOperator(a, l, X, Y)
            ls = LUSolver()
            solver = LinearFESolver(ls)
            xh = solve(solver, op)
            u_new, p_new = xh

            # ── Force extraction ──────────────────────────────────────────────
            if step % record_every == 0
                σ(u, p) = nu * (∇(u) + ∇(u)') - p * TensorValue(1.0, 0.0, 0.0, 1.0)
                Fx = sum(∫((σ(u_new, p_new) ⋅ n_cyl) ⊙ VectorValue(1.0, 0.0))dΓ)
                Fy = sum(∫((σ(u_new, p_new) ⋅ n_cyl) ⊙ VectorValue(0.0, 1.0))dΓ)
                Cd = 2 * Fx / (U_inlet^2 * D_phys)
                Cl = 2 * Fy / (U_inlet^2 * D_phys)
                push!(cd_series, Cd)
                push!(cl_series, Cl)
            end

            # Update history
            u_nm1 = u_n
            u_n = u_new

            te = time_ns()
            push!(step_times, Float64(te - ts) * 1e-3)

            step % 100 == 0 &&
                @printf("  FEM Re=%.0f  step=%d  Cd=%.4f\n",
                    cur_Re, step, isempty(cd_series) ? 0.0 : last(cd_series))

            if _converged(cd_series, cl_series, conv_window, conv_tol)
                @printf("  FEM Re=%.0f  converged at step %d  (%.0f samples)\n",
                    cur_Re, step, length(cl_series))
                break
            end
        end  # time loop

        t1 = time_ns()
        elapsed_total += t1 - t0

        if any(isnan.(cd_series)) || (!isempty(cl_series) && any(abs.(cl_series) .> 100))
            diverged = true
            @printf("  Re=%.0f diverged\n", cur_Re)
        else
            last_stable_Re = cur_Re
            @printf("  Re=%.0f stable  Cd_mean=%.4f\n", cur_Re, mean(cd_series))
        end

        append!(all_cd, cd_series)
        append!(all_cl, cl_series)
        append!(all_steps, step_times)

        diverged && break
    end

    return (cd=all_cd, cl=all_cl, steps=all_steps,
        elapsed_ns=elapsed_total, max_re=last_stable_Re)
end

end # module FEM_karman


const RE_VALUES = [300.0]

function main(;
    do_bench::Bool=false,
    do_cache::Bool=false,
    plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR,
    do_gif::Bool=false,
    sweep_re::Bool=false,
    run_fem::Bool=true,
)
    mkpath(plot_output_dir)

    if do_bench
        println("\n--- Benchmarking LBM (Re=150, 500 steps) ---")
        @time "Done benchmarking" LBM_karman.main(;
            n_steps=500, plot_every=typemax(Int), plot_mode="none", Re=150.0
        )
    end

    comparison_results = KarmanResult[]

    for Re in RE_VALUES
        println("\n" * "="^60)
        println("  Running all solvers at Re = $Re")
        println("="^60)

        # ── LBM ──────────────────────────────────────────────────────────────
        println("\n[LBM]")
        lbm_res = LBM_karman.main(;
            n_steps=20_000,
            plot_every=do_gif ? 200 : typemax(Int),
            plot_mode="vorticity",
            Re=Float64(Re),
            save_path=joinpath(plot_output_dir, "vorticity_Re$(round(Int, Re)).png"),
            do_gif=do_gif,
            gif_path=joinpath(plot_output_dir, "karman_Re$(round(Int, Re)).gif"),
        )
        # LBM returns nothing; build a stub result for comparison
        # (LBM has its own internal Cl tracking; wrap it here if available)
        push!(comparison_results, KarmanResult(
            "LBM", Float64(Re), 0.0, 0.0, 0.0, 0.0, Float64(Re)
        ))

        # ── FDM ──────────────────────────────────────────────────────────────
        println("\n[FDM]")
        fdm_res = FDM_karman.main(;
            Re=Float64(Re), record_every=20,
            plot_output_dir=plot_output_dir, sweep_re=sweep_re,
        )
        fdm_dt_phys = 0.4 * (KH_L / 220) / 0.1   # approx physical dt
        fdm_st = isempty(fdm_res.cl) ? 0.0 :
                 compute_strouhal(fdm_res.cl, fdm_dt_phys * 20, 0.1)
        push!(comparison_results, KarmanResult(
            "FDM",
            fdm_res.max_re,
            fdm_st,
            isempty(fdm_res.cd) ? 0.0 : mean(fdm_res.cd),
            isempty(fdm_res.cl) ? 0.0 : 0.5 * (maximum(fdm_res.cl) - minimum(fdm_res.cl)),
            isempty(fdm_res.steps) ? 0.0 : mean(fdm_res.steps),
            fdm_res.max_re,
        ))

        # ── FEM ──────────────────────────────────────────────────────────────
        if run_fem
            println("\n[FEM]")
            fem_res = FEM_karman.main(;
                Re=Float64(Re), record_every=20,
                plot_output_dir=plot_output_dir, sweep_re=sweep_re,
            )
            fem_dt_phys = 0.4 * 0.03 / 0.1   # dt = 0.4*h_cyl/U (default h_cyl=0.03)
            fem_st = isempty(fem_res.cl) ? 0.0 :
                     compute_strouhal(fem_res.cl, fem_dt_phys * 20, 0.1)
            push!(comparison_results, KarmanResult(
                "FEM",
                fem_res.max_re,
                fem_st,
                isempty(fem_res.cd) ? 0.0 : mean(fem_res.cd),
                isempty(fem_res.cl) ? 0.0 : 0.5 * (maximum(fem_res.cl) - minimum(fem_res.cl)),
                isempty(fem_res.steps) ? 0.0 : mean(fem_res.steps),
                fem_res.max_re,
            ))
        end
    end

    print_comparison(comparison_results, plot_output_dir)

    return
end

end # module
