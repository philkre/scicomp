# Usage
First cd to the correct folder
```bash
cd <repository>/laszlo/
```

## Installation
Run
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Run
```bash 
julia --project=. main.jl
```
Optional flag: bench to run benchmarks on different implementations of used algorithms
```bash
julia --project=. main.jl bench
```