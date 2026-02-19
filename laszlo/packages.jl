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
    "LaTeXStrings"
]

Pkg.add(dependencies)
