using Pkg

dependencies = [
    "IJulia",
    "Plots",
    "MPI",
    "SharedArrays",
    "LoopVectorization",
    "BenchmarkTools",
    "FFMPEG",
    "JLD2",
    "SpecialFunctions",
    "Printf",
    "LaTeXStrings",
    "ProgressMeter",
]

Pkg.add(dependencies)
