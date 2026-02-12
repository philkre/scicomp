using Pkg

dependencies = [
    "IJulia",
    "Plots",
    "MPI",
    "SharedArrays",
    "LoopVectorization",
    "BenchmarkTools",
    "FFMPEG"]

Pkg.add(dependencies)
