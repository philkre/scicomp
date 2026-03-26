import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

ENV["JULIA_CONDAPKG_RESOLVE"] = "false"