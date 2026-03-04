# Scientific Computing

Philipp Kreiter, Sijmen Kroon, Laszlo Schoonheid

## Assignment 1

This project contains numerical solvers and analysis scripts for classical scientific computing problems:
- 1D wave equation (Euler and Leapfrog time integration)
- 2D diffusion equation (time dependent)
- 2D Laplace steady-state problem (Jacobi, Gauss-Seidel, SOR)

The code is organized into reusable Julia modules for model kernels, simulation runners, data IO, and plotting.

## Setup

### 1. Requirements
- Julia 1.10+ (tested with Julia 1.12)

### 2. Install dependencies (idiomatic Julia)
From the project root, run:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Dependencies are pinned in:
- `Project.toml` (direct dependencies)
- `Manifest.toml` (fully resolved, reproducible environment)

### 3. Quick compile/load check

```bash
julia --project=. -e 'include("main.jl"); println("load ok")'
```

## Running

The main entrypoint is:

```bash
julia --project=. main.jl
```

`main.jl` contains separate functions for different tasks:
- `main_wave()`
- `main_diffusion()`
- `main_steadystate()`

Select which one runs by editing the `main()` function at the bottom of `main.jl`.

## Testing

Run the smoke test suite with:

```bash
julia --project=. test/runtests.jl
```

Current smoke coverage includes:
- module/load sanity (`main.jl` + all modules)
- wave runs (`run_wave`, `run_wave_1b`)
- diffusion runs + HDF5 IO (`run_diffusion`, `DataIO.load_output`)
- steady-state solvers (`jacobi`, `gauss-seidel`, `sor`)
- omega sweep outputs/masks (`optimise_omega`)
- panel plot generation (`plot_omega_sweep_panels`)

## Folder Structure

- `main.jl`
  - top-level orchestration and scenario selection
- `model.jl`
  - numerical kernels (PDE stencil updates, analytical profile helper)
- `sim.jl`
  - run functions for wave, diffusion, steady-state, and omega sweep
- `data.jl`
  - HDF5 read/write utilities
- `plot.jl`
  - plotting and post-processing utilities
- `test/`
  - smoke test coverage
- `output/`
  - generated artifacts
  - `output/data/`: numerical data files (`.h5`)
  - `output/img/`: images and videos (`.png`, `.mp4`)

## Notes

- Omega optimisation for SOR supports a constrained `(omega, N)` region and convergence masks.
- Plot outputs are written automatically to `output/img/`.
