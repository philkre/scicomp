# Plotting parameters
get_heatmap_ticks = (N::Int64, L::Float64) -> (1:N/4:N+1, 0:L/4:L)
get_heatmap_lims = (N::Int64) -> (1, N + 1)
get_heatmap_kwargs = (N::Int64, L::Float64) -> Dict(
    :aspect_ratio => 1,
    :xlabel => "x",
    :ylabel => "y",
    :xticks => get_heatmap_ticks(N, L),
    :yticks => get_heatmap_ticks(N, L),
    :xlims => get_heatmap_lims(N),
    :ylims => get_heatmap_lims(N),
    :dpi => 150,
    :legend => true,
)