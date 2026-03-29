# FDM + FEM Kármán Vortex Street Solvers — Design Spec
_Date: 2026-03-29_

## Goal

Add two new solvers — FDM (finite difference) and FEM (finite element) — to `src/ass_3/assignment_3_1.jl` alongside the existing LBM solver. All three solve the same incompressible Navier-Stokes problem (Figure 4 geometry) and are compared on Strouhal number, Cd, Cl amplitude, wall-clock time per step, and maximum stable Re.

---

## Geometry (shared, all solvers)

Physical domain from Assignment Figure 4:
- Width: L = 2.2 m
- Height: H = 0.31 m  (0.16 m above + 0.15 m below cylinder centre)
- Cylinder centre: (cx, cy) = (0.15 m, 0.15 m)
- Cylinder diameter: D = 0.1 m, radius r = 0.05 m
- Inlet velocity: U_inlet (set per Re; LBM uses U_inlet=0.12 lattice units, FDM/FEM use physical units directly)

---

## File Structure

```
src/ass_3/assignment_3_1.jl
  └── module Assignment_3_1
        ├── include("../../helpers/karman_helpers.jl")
        ├── module LBM_karman    (existing BGK solver — unchanged)
        ├── module FDM_karman    (new — MAC staggered, fractional step)
        └── module FEM_karman    (new — Gridap.jl, Taylor-Hood P2/P1)

src/helpers/karman_helpers.jl   (new)
```

Dependencies to add to `Project.toml`: `Gridap`, `GridapGmsh`, `Gmsh`.

---

## `karman_helpers.jl`

Shared constants and utilities used by all three solvers.

**Constants:**
```julia
const KH_L    = 2.2      # domain length (m)
const KH_H    = 0.31     # domain height (m)
const KH_D    = 0.1      # cylinder diameter (m)
const KH_CX   = 0.15     # cylinder centre x (m)
const KH_CY   = 0.15     # cylinder centre y (m)
```

**Functions:**
- `compute_strouhal(cl_series, dt_phys) → St`  — FFT peak of mean-subtracted Cl; `dt_phys` in seconds
- `print_comparison(results)` — prints and saves `plots/ass_3/comparison.txt` with columns: Solver, St, Cd, Cl_amp, t/step, Max Re
- `save_velocity_plot(ux, uy, obs_or_nothing, Re, solver_name, output_dir)` — heatmap of speed field, saves `<solver>_max_Re<N>.png`

---

## Solver Interface

Each sub-module exposes one public function:

```julia
main(;
    Re::Float64          = 100.0,
    n_steps::Int         = 30_000,
    record_every::Int    = 20,
    plot_output_dir::String = "plots/ass_3",
    sweep_re::Bool       = false,
) → (cd, cl, steps, elapsed_ns, max_re)
```

- `cd`, `cl`: `Vector{Float64}` of sampled Cd/Cl values
- `steps`: corresponding LBM/FDM/FEM step indices
- `elapsed_ns`: total wall-clock time in nanoseconds
- `max_re`: last stable Re (equal to `Re` if `sweep_re=false`)

When `sweep_re=true`, the solver runs Re = 100, 200, 300, 500, 1000 in sequence, stopping at the first divergence (`any(isnan, cd)` or `maximum(abs.(cl)) > 100`). Saves a velocity plot at the last stable Re.

---

## FDM Solver (`module FDM_karman`)

### Grid

MAC staggered grid:
- `u[i,j]` (x-velocity) defined at `(i+½, j)` — size `(Nx+1) × Ny`
- `v[i,j]` (y-velocity) defined at `(i, j+½)` — size `Nx × (Ny+1)`
- `p[i,j]` (pressure) defined at cell centres `(i, j)` — size `Nx × Ny`

Default resolution: `Nx=220, Ny=62` (dx ≈ 0.01 m). Configurable.

### Cylinder

Immersed Boundary: cells whose centre lies inside `(x-cx)²+(y-cy)² ≤ r²` are flagged as obstacle. Velocity forced to zero after each stage. Matches LBM staircase geometry for fair St/Cd comparison.

### Timestep (fractional step, 3 stages)

1. **Intermediate velocity `u*`** — explicit Adams-Bashforth 2nd-order advection + explicit Laplacian diffusion:
   ```
   u* = u^n + dt * (-(3/2)(u·∇u)^n + (1/2)(u·∇u)^{n-1} + ν ∇²u^n)
   ```
   Time step limited by CFL: `dt = 0.4 * dx / U_max`.

2. **Pressure Poisson** — solve `∇²p = (1/dt) ∇·u*` using SOR iteration (convergence criterion: `max|residual| < 1e-5`).

3. **Velocity correction** — `u = u* − dt ∇p`. Enforce IB and wall BCs.

### Boundary Conditions

- Inlet (x=0): `u = U_inlet`, `v = 0`
- Outlet (x=L): zero-gradient `∂u/∂x = 0`, `p = 0`
- Top/bottom walls: no-slip `u = v = 0`
- Cylinder: IB zero-velocity enforcement

### Force Extraction

Sum pressure and viscous contributions on all fluid cells adjacent to cylinder obstacle cells:
```
Fx = Σ (p·nx - ν·∂u/∂n) · dA
Cd = 2 Fx / (ρ U_inlet² D)
```

---

## FEM Solver (`module FEM_karman`)

### Mesh

Generated once by Gmsh: structured triangular mesh of the 2.2×0.31 rectangle with a circular hole of radius 0.05 m centred at (0.15, 0.15). Mesh size: `h=0.008` near cylinder, `h=0.02` far field. Saved to `plots/ass_3/karman_mesh.msh` and reused across runs.

### Elements

Taylor-Hood P2/P1 (quadratic velocity, linear pressure) — inf-sup stable, no pressure stabilisation required.

### Formulation

Weak form of incompressible NS time-marched with BDF2. At each timestep one nonlinear system is solved via Newton iteration (3–5 iterations typical). Gridap assembles the sparse system; `LUSolver` (backslash) handles the linear solve.

### Boundary Conditions

All imposed strongly (Dirichlet tags from Gmsh):
- Inlet: `u = (U_inlet, 0)`
- Cylinder surface + top/bottom walls: `u = (0, 0)`
- Outlet: stress-free (do-nothing) Neumann — no tag needed, natural BC

### Force Extraction

Integrate stress tensor over cylinder boundary:
```
F = ∫_Γ (-p I + ν(∇u + ∇uᵀ)) · n dΓ
```
via Gridap's `∫(n_Γ ⊙ σ)dΓ`. Most physically accurate Cd/Cl of the three methods.

---

## Outer `main()` in `Assignment_3_1`

```julia
function main(;
    do_bench=false,
    do_cache=false,
    plot_output_dir=DEFAULT_PLOT_OUTPUT_DIR,
    sweep_re=false,
)
```

Runs all three solvers at Re=100 (and Re sweep if `sweep_re=true`), collects results, calls `print_comparison`. Also calls existing LBM single-Re run to generate vorticity plot.

---

## Comparison Output

**Metrics table** (stdout + `plots/ass_3/comparison.txt`):

```
Solver │  St    │  Cd   │ Cl_amp │ t/step │ Max Re
───────┼────────┼───────┼────────┼────────┼───────
LBM    │  ?     │  ?    │   ?    │   ?    │   ?
FDM    │  ?     │  ?    │   ?    │   ?    │   ?
FEM    │  ?     │  ?    │   ?    │   ?    │   ?
```

**Velocity plots at max stable Re** per solver:
- `plots/ass_3/LBM_max_Re<N>.png`
- `plots/ass_3/FDM_max_Re<N>.png`
- `plots/ass_3/FEM_max_Re<N>.png`

---

## Testing

- **Unit tests** in `test/ass_3/test_lbm.jl` (existing) — unchanged
- **FDM smoke test**: 500 steps at Re=100, check `isfinite(mean(cd))` and `St` in `[0.12, 0.22]`
- **FEM smoke test**: 200 steps at Re=100, same checks
- No Strouhal regression test for FDM/FEM (too slow for CI); manual validation run required

---

## Self-Review Notes

- Geometry constants are defined once in `karman_helpers.jl` and imported by all three modules — no duplication.
- FDM uses physical units (m, s) throughout; LBM uses lattice units. The `compute_strouhal` function in helpers takes `dt_phys` in seconds so all three use the same post-processing code.
- FEM mesh is generated once and cached to disk — subsequent runs skip Gmsh entirely.
- `sweep_re=true` is opt-in to keep the default run fast (~2 min for all three at Re=100).
