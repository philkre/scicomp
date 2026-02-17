# two-d time dependent diffusion equation
using HDF5
using Plots
using Statistics
using ProgressMeter
using SpecialFunctions
using Printf
using LinearAlgebra

function diffusion2d!(c::Array{Float64,2}, D::Float64, dx::Float64, dt::Float64)
    """
    Discretisation of the 2D diffusion equation: dC / dt = D * (d^2 C / dx^2 + d^2 C / dy^2)
    shape: c[x,y] -> c[i,j]
    """
    # boundaries
    c_curr = copy(c)
    max_i = size(c, 1)
    max_j = size(c, 2)

    @inbounds for j in 2:max_j-1
        for i in 1:max_i
            # periodic boundary condiitions
            ip = (i == max_i) ? 2 : i + 1
            im = (i == 1) ? max_i - 1 : i - 1

            c[i, j] = c_curr[i, j] + D * dt / (dx * dx) * (
                c_curr[ip, j] + c_curr[im, j] +
                c_curr[i, j+1] + c_curr[i, j-1] -
                4 * c_curr[i, j]
            )
        end
    end

    # y direction bounds
    c[:, 1] .= 0
    c[:, end] .= 1

    # x direction bounds enforced
    c[1, :] .= c[end, :]
end

function write_out(c::Array{Float64,2}, filepath::String, step::Int64, dx::Float64, dy::Float64; t=nothing)
    """
    Writes the data in c to a HDF5 file at the specified filepath. 
    """

    Nx, Ny = size(c)

    h5open(filepath, "cw") do f
        # create file if it does not exist
        if !haskey(f, "c")
            # store c in format (Nx, Ny, Nt)
            create_dataset(f, "c", eltype(c), dataspace((Nx, Ny, 0);
                    max_dims=(Nx, Ny, -1)), chunk=(Nx, Ny, 1))

            # steps as 1D dataset
            create_dataset(f, "step", Int, dataspace((0,);
                    max_dims=(-1,)), chunk=(1024,))

            if t !== nothing
                create_dataset(f, "time", eltype(t), dataspace((0,);
                        max_dims=(-1,)), chunk=(1024,))
            end

            # store grid spacing ones
            attrs = attributes(f)
            attrs["dx"] = float(dx)
            attrs["dy"] = float(dy)
        else
            # ensuer grid size matches
            dset = f["c"]
            sx, sy, _ = size(dset)
            if (sx, sy) != (Nx, Ny)
                error("Grid size mismatch: expected ($Nx, $Ny), got ($sx, $sy)")
            end
        end

        # append new slice
        dset_c = f["c"]
        _, _, nt = size(dset_c)
        new_nt = nt + 1

        HDF5.set_extent_dims(dset_c, (Nx, Ny, new_nt))
        dset_c[:, :, new_nt] = c

        # append step
        dset_step = f["step"]
        HDF5.set_extent_dims(dset_step, (new_nt,))
        dset_step[new_nt] = step

        # Append time if present + requested
        if t !== nothing && haskey(f, "time")
            dset_t = f["time"]
            HDF5.set_extent_dims(dset_t, (new_nt,))
            dset_t[new_nt] = float(t)
        end
    end

    return nothing
end

function run(D::Float64, N::Int64, dy::Float64, dx::Float64, dt::Float64, steps::Int64, write_interval::Int64; timing::Bool=false, progress::Bool=false, filepath::String="output.h5")
    """
    Runs the diffusion simulation for a given set of parameters.
    - D: diffusion coefficient 
    - N: number of grid points in x direction 
    - dy: grid spacing in y direction 
    - dx: grid spacing in x direction 
    - dt: time step size 
    - steps: total number of time steps to simulate 
    - write_interval: interval at which to write output (e.g., every 10 steps)
    """

    # check stability
    stab = 4 * dt * D / (dx * dx)
    @assert stab <= 1 "Stability condition violated: 4*D*dt/dx^2 must be <= 1, is $stab"

    c = zeros(N, N)
    # init boundaries
    c[:, 1] .= 0
    c[:, end] .= 1

    p = progress ? Progress(steps; desc="Diffusion", barlen=30) : nothing

    # delete h5 file if it exists 
    if isfile(filepath)
        rm(filepath)
    end

    elapsed = @elapsed begin
        for step in 1:steps
            diffusion2d!(c, D, dx, dt)
            if step % write_interval == 0
                # write out c to jsonl
                write_out(c, filepath, step, dx, dy; t=step * dt)
            end
            if progress
                next!(p)
            end
        end
    end

    if timing
        println("run completed in $(round(elapsed; digits=6)) s")
    end
end

function plot_animation(filepath::String="output.h5"; fps::Int64=30, filename::String="diffusion_anim.mp4", max_frames::Int64=300, stride::Int64=nothing)
    """
    Loads diffusion data from an HDF5 file and creates an animated heatmap over time.
    """
    h5open(filepath, "r") do f
        c = read(f["c"])
        dx = read(attributes(f)["dx"])
        dy = read(attributes(f)["dy"])
        nt = size(c, 3)

        x = (0:size(c, 1)-1) .* dx
        y = (0:size(c, 2)-1) .* dy
        steps = haskey(f, "step") ? read(f["step"]) : collect(1:nt)

        local_stride = stride === nothing ? max(1, cld(nt, max_frames)) : stride
        frame_indices = 1:local_stride:nt

        anim = @animate for t in frame_indices
            heatmap(
                x,
                y,
                c[:, :, t]',
                xlabel="x",
                ylabel="y",
                title="Diffusion 2D — step $(steps[t])",
                aspect_ratio=1,
                color=:viridis,
                clims=(0, 1),
                xlims=(x[1], x[end]),
                ylims=(y[1], y[end]),
            )
        end

        mp4(anim, filename, fps=fps)
    end
end

function analytical_profile_series(x::AbstractVector{<:Real}, t::Real, D::Real, L::Real; n_terms::Integer=200)
    """Truncated image-series solution:
    c(x,t) = sum_{i=0}^∞ [erfc((L - x + 2iL)/(2*sqrt(Dt))) - erfc((L + x + 2iL)/(2*sqrt(Dt)))]"""

    # ensure t > 0
    if t <= 0
        throw(ArgumentError("t must be > 0 for analytical_profile_series"))
    end
    # precompute denominator
    denom = 2 * sqrt(D * t)
    c = zeros(Float64, length(x))

    # sum series terms
    for i in 0:n_terms
        c .+= erfc.((L .- x .+ 2i * L) ./ denom) .- erfc.((L .+ x .+ 2i * L) ./ denom)
    end

    return c
end

function plot_profiles(filepath::String="output.h5", output::String="profiles.png";
    times::Vector{Float64}=Float64[],
    x_index::Int64=0,
    average_x::Bool=true,
    D::Float64=1.0,
    n_terms::Int64=200)
    """
    Plots concentration as a function of y for selected time points.
    If a time dataset exists, `times` can be given in physical time units and
    the nearest stored times will be used. Otherwise, `times` is interpreted
    as frame indices (1-based). Defaults to all frames if none provided.
    """
    h5open(filepath, "r") do f
        # read data
        c = read(f["c"])
        dy = read(attributes(f)["dy"])
        y = (0:size(c, 2)-1) .* dy
        L = y[end]

        # get times if available
        has_time = haskey(f, "time")
        tvals = has_time ? read(f["time"]) : nothing

        # normalize to 3D for unified handling
        c3 = ndims(c) == 2 ? reshape(c, size(c, 1), size(c, 2), 1) : c
        nt = size(c3, 3)

        # determine indices to plot
        if isempty(times)
            indices = 1:nt
        else
            if has_time
                indices = [findmin(abs.(tvals .- t))[2] for t in times]
            else
                indices = Int.(times)
            end
        end

        # plot profile for each selectd point in time
        p = plot(dpi=300)
        for (k, idx) in enumerate(indices)

            # slice
            slice = c3[:, :, idx]

            # get profile (average over x or slice at x_index)
            prof = average_x ? vec(mean(slice, dims=1)) :
                   slice[isnothing(x_index) ? 1 : x_index, :]
            label = has_time ? "t=$(tvals[idx])" : "frame=$idx"
            plot!(p, y, prof, label=label, seriescolor=k)

            # add analytical solution if time data is available and t > 0
            if has_time
                t = tvals[idx]
                if t > 0
                    c_analytical = analytical_profile_series(y, t, D, L; n_terms=n_terms)
                    plot!(p, y, c_analytical, linestyle=:dash, label="analytical t=$(round(t; digits=4))", seriescolor=k)
                end
            end
        end

        # print error between numerical and analytical at final time if possible
        if has_time && !isempty(times)
            # last idx
            final_idx = indices[end]
            final_t = tvals[final_idx]

            if final_t > 0
                # compute numerical and analytical profiles at final time
                numerical_final = average_x ? vec(mean(c3[:, :, final_idx], dims=1)) : c3[isnothing(x_index) ? 1 : x_index, :, final_idx]
                analytical_final = analytical_profile_series(y, final_t, D, L; n_terms=n_terms)

                # caculate and print relative L2 error
                error = norm(numerical_final - analytical_final) / norm(analytical_final)
                println("Relative L2 error at final time (t=$(final_t)): $(round(error; sigdigits=3))")
            end
        end

        xlabel!(p, "y")
        ylabel!(p, "concentration")
        title!(p, "Concentration vs y")
        display(p)
        savefig(output)
    end
end

function plot_2d_concentration(
    filepath::String="output.h5",
    output::String="concentration.png",
    timesteps::Vector{Float64}=[1.0])
    """
    Plots a 2D heatmap of concentration at defined timesteps
    """

    h5open(filepath, "r") do f
        # read data
        c = read(f["c"])
        dx = read(attributes(f)["dx"])
        dy = read(attributes(f)["dy"])
        x = (0:size(c, 1)-1) .* dx
        y = (0:size(c, 2)-1) .* dy

        # timesteps to indices
        tvals = read(f["time"])
        timesteps_idx = [findmin(abs.(tvals .- t))[2] for t in timesteps]

        subplots = []
        for (i, time_index) in enumerate(timesteps_idx)
            # slice timestep
            slice = c[:, :, time_index]
            # get time for title
            time = timesteps[i]
            # plot
            p = heatmap(x, y, slice', xlabel="x", ylabel="y", title="Concentration at time $time", aspect_ratio=1, color=:viridis)
            push!(subplots, p)
        end
        # subplots
        plot(subplots..., layout=(ceil(Int64, length(subplots) / 2), 2), size=(1800, 1200), dpi=300)
        savefig(output)
    end
end


function main()
    D = 1.0
    Lx = 1.0
    Ly = 1.0
    N = 100
    dy = Ly / (N - 1)
    dx = Lx / (N - 1)
    dt = 0.00001
    T = 1
    steps = ceil(Int, T / dt)
    write_interval = 1

    print("Discretisation: dx=$(@sprintf("%.3f", dx)), dy=$(@sprintf("%.3f", dy)), dt=$dt, steps=$steps\n")

    #run(D, N, dy, dx, dt, steps, write_interval; timing=true, progress=true)
    #plot_animation("output.h5"; fps=30, filename="diffusion_anim.mp4")
    #plot_profiles("output.h5", "profiles.png"; times=[0.001, 0.01, 0.1, 1.0], average_x=true, D=D)
    #plot_2d_concentration("output.h5", "concentration.png", [0.0, 0.001, 0.01, 0.1, 1.0])
end

main()
