module GrayScottUtil

using Plots: plot, heatmap, @animate, gif
using Printf: @sprintf


include("get_heatmap_kwargs.jl")


"""
    plot_gray_scott_state(u::Matrix{Float64}, v::Matrix{Float64}, t::Float64;
                         title_prefix::String="Gray-Scott")

Create a plot showing both U and V concentrations side by side.

# Arguments
- `u::Matrix{Float64}`: U concentration field
- `v::Matrix{Float64}`: V concentration field
- `t::Float64`: Current time
- `title_prefix::String`: Prefix for plot title

# Returns
- Plot object with two heatmaps
"""
function plot_gray_scott_state(
    u::Matrix{Float64},
    v::Matrix{Float64},
    t::Float64
    ;
    L::Float64=1.0,
    title_prefix::String="Gray-Scott")
    N = size(u, 1)
    heatmap_kwargs = get_heatmap_kwargs(N, L)

    # Plot U
    p1 = heatmap(
        u';
        heatmap_kwargs...,
        c=:viridis,
        title="U at t=$(round(t, digits=1))",
        clims=(0, 1)
    )
    # Plot V
    p2 = heatmap(
        v';
        heatmap_kwargs...,
        c=:plasma,
        title="V at t=$(round(t, digits=1))",
        clims=(0, maximum(v))
    )

    # Combine plots
    plot(
        p1, p2,
        layout=(1, 2),
        size=(1200, 500),
        plot_title="$title_prefix (f=$(@sprintf("%.4f", 0.035)), k=$(@sprintf("%.4f", 0.060)))"
    )
end


end # module