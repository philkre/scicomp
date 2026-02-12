# two-d time dependent diffusion equation
using HDF5
using Plots
using Statistics
using ProgressMeter

function diffusion2d!(c, D, dx, dt)
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

function write_out(c, filepath, step, dx, dy; t=nothing)
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

function run(D, N, dy, dx, dt, steps, write_interval; timing=false, progress=false, filepath="output.h5")
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

function plot_animation(filepath="output.h5"; fps=30, filename="diffusion_anim.mp4", max_frames=300, stride=nothing)
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

function plot_profiles(filepath="output.h5", output="profiles.png"; times=nothing, x_index=nothing, average_x=true)
    """
    Plots concentration as a function of y for selected time points.
    If a time dataset exists, `times` can be given in physical time units and
    the nearest stored times will be used. Otherwise, `times` is interpreted
    as frame indices (1-based). Defaults to all frames if none provided.
    """
    h5open(filepath, "r") do f
        c = read(f["c"])
        dy = read(attributes(f)["dy"])
        y = (0:size(c, 2)-1) .* dy

        has_time = haskey(f, "time")
        tvals = has_time ? read(f["time"]) : nothing

        # normalize to 3D for unified handling
        c3 = ndims(c) == 2 ? reshape(c, size(c, 1), size(c, 2), 1) : c
        nt = size(c3, 3)

        if times === nothing
            indices = 1:nt
        else
            if has_time
                indices = [findmin(abs.(tvals .- t))[2] for t in times]
            else
                indices = Int.(times)
            end
        end

        p = plot()
        for idx in indices
            slice = c3[:, :, idx]
            prof = average_x ? vec(mean(slice, dims=1)) :
                   slice[isnothing(x_index) ? 1 : x_index, :]
            label = has_time ? "t=$(tvals[idx])" : "frame=$idx"
            plot!(p, y, prof, label=label)
        end

        xlabel!(p, "y")
        ylabel!(p, "concentration")
        title!(p, "Concentration vs y")
        display(p)
        savefig(output)
    end
end


function main()
    D = 0.1
    N = 100
    dy = 0.01
    dx = 0.01
    dt = 0.000001
    T = 1
    steps = Int(T / dt)
    write_interval = Int(steps / 10)

    # run(D, N, dy, dx, dt, steps, write_interval; timing=true, progress=true)
    # plot_animation("output.h5"; fps=30, filename="diffusion_anim.mp4")
    plot_profiles("output.h5"; times=[0.001, 0.01, 0.1, 1.0], average_x=true)
end

main()
