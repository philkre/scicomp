import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include("model.jl")
include("data.jl")
include("plot.jl")

using .Model: laplace_sor!
using .Plotting: plot_steadystate
using StatsBase
using Plots: @animate, heatmap, contour!, mp4

function update_mask!(c, mask, nu)

    # get all growth candidates
    mask_candidates = Tuple{Int,Int}[]
    candidate_weights = Float64[]
    candidate_seen = falses(size(mask))

    # iterate through all free sites and check if they are adjacent to occupied sites
    for I in findall(mask .== 0.0)
        # occupied sites
        i, j = Tuple(I)
        # check neighbors        
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ni, nj = i + di, j + dj
            if 1 <= ni <= size(mask, 1) && 1 <= nj <= size(mask, 2)
                if mask[ni, nj] == 1.0 && !candidate_seen[ni, nj]
                    # candidate for growth
                    push!(mask_candidates, (ni, nj))
                    candidate_seen[ni, nj] = true
                    push!(candidate_weights, max(c[ni, nj], 0.0)^nu)
                end
            end
        end
    end

    # choose one candidate to grow based on probabilities
    if !isempty(mask_candidates)
        wsum = sum(candidate_weights)
        chosen_index = if isfinite(wsum) && wsum > 0.0
            sample(1:length(mask_candidates), Weights(candidate_weights))
        else
            rand(1:length(mask_candidates))
        end
        chosen_site = mask_candidates[chosen_index]
        mask[chosen_site...] = 0.0 # mark as occupied
    end

end

function run_dla(
    N::Int,
    steps::Int,
    nu::Float64,
    omega::Float64
)
    # init concentration grid
    c = zeros(N, N)
    c[:, 1] .= 0.0
    c[:, end] .= 1.0

    # mask for DLA process (0 = occupied, 1 = free)
    m = ones(N, N)
    init_pos = [ceil(Int, N / 2), 1]
    m[init_pos...] = 0.0

    cs = Matrix{Float64}[]
    masks = Matrix{Float64}[]
    for _ in 1:100
        # mask update with DLA rules
        update_mask!(c, m, nu)
        sink_indices = findall(m .== 0.0)
        for step in 1:steps

            # diffusion step with new mask
            laplace_sor!(c, omega; sink_indices=sink_indices)
        end

        # store results for plotting
        push!(cs, copy(c))
        push!(masks, copy(m))
    end
    return Dict("cs" => cs, "masks" => masks)
end

function animate_dla(results; filename="philipp/output/img/dla_animation.mp4", fps=30, max_frames=300)

    cs = results["cs"]
    masks = results["masks"]
    nt = length(cs)
    stride = max(1, cld(nt, max_frames))

    anim = @animate for i in 1:stride:nt
        c = cs[i]
        sink = masks[i] .== 0.0

        p = heatmap(
            c',
            aspect_ratio=1,
            color=:viridis,
            clims=(0, 1),          # avoid color flicker
            title="DLA step $i",
            xlabel="x",
            ylabel="y",
        )
        contour!(p, Float64.(sink)', levels=[0.5], color=:white, linewidth=1.5, label=false)
    end

    mkpath(dirname(filename))
    mp4(anim, filename; fps=fps)
    return filename
end


function main_dla()
    N = 100
    steps = 1000
    nu = 1.0
    omega = 1.8

    results = run_dla(N, steps, nu, omega)

    plot_steadystate(
        results["cs"][end],
        "philipp/output/img/dla_concentration.png";
        sink_indices=findall(results["masks"][end] .== 0.0),
    )
end

main_dla()
