# Assignment 1

## Usage

### Installation
Run
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Run
```bash 
julia --project=. assignment_1.jl
```
Optional flag: `bench` to run benchmarks on different implementations of used algorithms
```bash
julia --project=. assignment_1.jl bench
```

Optional flag: `gif` to create animations of the simulations
```bash
julia --project=. assignment_1.jl gif
```