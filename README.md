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

# Assignment 2

## Usage

### Installation
Run
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Run
Note: When not ran on macOS multiple metal.jl errors will be printed, but these are just used for GPU optimization on macOS and this will not influence the results  
```bash 
julia --project=. assignment_2.jl
```
Optional flags: run this command to see optional flags for assignment 2
```bash
julia --project=. assignment_2.jl -h
```
