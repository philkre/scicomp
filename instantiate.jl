import Pkg
Pkg.activate(@__DIR__)

# Add local Helpers package as a dev dependency
Pkg.develop(Pkg.PackageSpec(path=joinpath(@__DIR__, "src", "helpers")))

Pkg.instantiate()

ENV["JULIA_CONDAPKG_RESOLVE"] = "false"