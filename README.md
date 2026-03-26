# Installation

Run
```bash
julia instantiate.jl
```

# Assignment 1

## Usage

### Run
```bash 
julia assignment_1.jl
```
Optional flag: `bench` to run benchmarks on different implementations of used algorithms
```bash
julia assignment_1.jl bench
```

Optional flag: `gif` to create animations of the simulations
```bash
julia assignment_1.jl gif
```

# Assignment 2

## Usage

### Run
Note: When not ran on macOS multiple metal.jl errors will be printed, but these are just used for GPU optimization on macOS and this will not influence the results  
```bash 
julia assignment_2.jl
```
Optional flags: run this command to see optional flags for assignment 2
```bash
julia assignment_2.jl -h
```
This executes the base DLA and Gray-Scott models. Computationally plots where execution times are long will be generated from the saved .csv files via

```bash
julia assignment_2.jl --rerender-all-csv-plots -p plots/ass_2

```
The large grid scaling examples for instance took more than twentyfour hours to run. Study the optional flags to execute these experiments individually.
