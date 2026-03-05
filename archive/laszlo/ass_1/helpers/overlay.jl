function _overlay_image!(
    p,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    sink_mask::AbstractMatrix{Bool};
    subplot_idx::Int=1,
    image_path::String="input/sink",
    scale::Float64=0.55,
)
    candidates = splitext(image_path)[2] == "" ? [image_path * ".png", image_path] : [image_path]
    resolved_path = ""
    for cand in candidates
        path = isabspath(cand) ? cand : normpath(joinpath(pwd(), cand))
        if isfile(path)
            resolved_path = path
            break
        end
    end
    if isempty(resolved_path)
        @warn "Failed to locate sink image, skipping sink overlay." image_path
        return nothing
    end

    img = try
        load(resolved_path)
    catch err
        @warn "Failed to load sink image, skipping sink overlay." resolved_path exception = (err, catch_backtrace())
        return nothing
    end
    img2d = ndims(img) == 2 ? img : img[:, :, 1]

    # Crop to non-background content so the sink is visible at subplot scale.
    alpha_mask = trues(size(img2d))
    if eltype(img2d) <: Colorant
        alpha_mask = [alpha(px) > 0.02 for px in img2d]
        # Fallback for files with opaque black background: keep bright pixels.
        if !any(alpha_mask)
            alpha_mask = [red(px) + green(px) + blue(px) > 0.05 for px in img2d]
        end
    else
        alpha_mask = img2d .> 0.02
    end

    if any(alpha_mask)
        rows = findall(vec(any(alpha_mask, dims=2)))
        cols = findall(vec(any(alpha_mask, dims=1)))
        img2d = img2d[minimum(rows):maximum(rows), minimum(cols):maximum(cols)]
    end

    sink_idxs = findall(sink_mask)
    if isempty(sink_idxs)
        return nothing
    end

    # Center image on mask centroid.
    xs = [x[I[1]] for I in sink_idxs]
    ys = [y[I[2]] for I in sink_idxs]
    cx = mean(xs)
    cy = mean(ys)
    # Use fixed image size in axis units to keep overlays consistent across mask shapes.
    xr = maximum(x) - minimum(x)
    yr = maximum(y) - minimum(y)
    hx = max(eps(Float64), 0.5 * xr * scale)
    hy = max(eps(Float64), 0.5 * yr * scale)

    # plot image
    plot!(
        p,
        [cx - hx, cx + hx],
        [cy - hy, cy + hy],
        img2d;
        subplot=subplot_idx,
        seriestype=:image,
        yflip=false,
        label=false,
    )
    return nothing
end
