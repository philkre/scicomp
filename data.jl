module DataIO

using HDF5

export write_out!, load_output

function write_out!(
    c::Matrix{Float64},
    filepath::String,
    step::Int,
    dx::Float64,
    dy::Float64;
    t=nothing,
)
    """
    Appends concentration data to an HDF5 file.
    """
    mkpath(dirname(filepath))
    Nx, Ny = size(c)

    h5open(filepath, "cw") do f
        if !haskey(f, "c")
            create_dataset(
                f,
                "c",
                eltype(c),
                dataspace((Nx, Ny, 0); max_dims=(Nx, Ny, -1)),
                chunk=(Nx, Ny, 1),
            )

            create_dataset(
                f,
                "step",
                Int,
                dataspace((0,); max_dims=(-1,)),
                chunk=(1024,),
            )

            if t !== nothing
                create_dataset(
                    f,
                    "time",
                    eltype(t),
                    dataspace((0,); max_dims=(-1,)),
                    chunk=(1024,),
                )
            end

            attrs = attributes(f)
            attrs["dx"] = float(dx)
            attrs["dy"] = float(dy)
        else
            dset = f["c"]
            sx, sy, _ = size(dset)
            if (sx, sy) != (Nx, Ny)
                error("Grid size mismatch: expected ($Nx, $Ny), got ($sx, $sy)")
            end
        end

        dset_c = f["c"]
        _, _, nt = size(dset_c)
        new_nt = nt + 1

        HDF5.set_extent_dims(dset_c, (Nx, Ny, new_nt))
        dset_c[:, :, new_nt] = c

        dset_step = f["step"]
        HDF5.set_extent_dims(dset_step, (new_nt,))
        dset_step[new_nt] = step

        if t !== nothing && haskey(f, "time")
            dset_t = f["time"]
            HDF5.set_extent_dims(dset_t, (new_nt,))
            dset_t[new_nt] = float(t)
        end
    end

    return nothing
end

function load_output(filepath::String)
    """
    Loads diffusion data and metadata from an HDF5 file.
    """
    h5open(filepath, "r") do f
        c = read(f["c"])
        dx = read(attributes(f)["dx"])
        dy = read(attributes(f)["dy"])
        steps = haskey(f, "step") ? read(f["step"]) : collect(1:size(c, 3))
        times = haskey(f, "time") ? read(f["time"]) : nothing
        return c, dx, dy, steps, times
    end
end

end
