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
    "LaTeXStrings"
]

Pkg.add(dependencies)
