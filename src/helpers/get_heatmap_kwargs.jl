# Plotting parameters

"""
    get_heatmap_ticks(N::Int64, L::Float64)

Generate tick positions for heatmap axes.

# Arguments
- `N::Int64`: Number of grid points
- `L::Float64`: Physical domain length

# Returns
- `Tuple`: Tuple of (tick_positions, tick_labels) for both x and y axes
"""
get_heatmap_ticks = (N::Int64, L::Float64) -> (1:N/4:N+1, 0:L/4:L)

"""
    get_heatmap_lims(N::Int64)

Generate axis limits for heatmap plots.

# Arguments
- `N::Int64`: Number of grid points

# Returns
- `Tuple{Int, Int}`: Tuple of (lower_limit, upper_limit)
"""
get_heatmap_lims = (N::Int64) -> (1, N + 1)

"""
    get_heatmap_kwargs(N::Int64, L::Float64)

Generate a dictionary of keyword arguments for heatmap plotting with standard settings.

# Arguments
- `N::Int64`: Number of grid points
- `L::Float64`: Physical domain length

# Returns
- `Dict`: Dictionary containing plot settings including aspect ratio, labels, ticks, limits, dpi, and legend
"""
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

"""
    get_heatmap_kwargs(Nx::Int64, Ny::Int64, Lx::Float64, Ly::Float64)

Generate a dictionary of keyword arguments for heatmap plotting with standard settings.

# Arguments
- `Nx::Int64`: Number of grid points in x direction
- `Ny::Int64`: Number of grid points in y direction
- `Lx::Float64`: Physical domain length in x direction
- `Ly::Float64`: Physical domain length in y direction
- `L::Float64`: Physical domain length

# Returns
- `Dict`: Dictionary containing plot settings including aspect ratio, labels, ticks, limits, dpi, and legend
"""
get_heatmap_kwargs = (Nx::Int64, Ny::Int64, Lx::Float64, Ly::Float64) -> Dict(
    :aspect_ratio => 1,
    :xlabel => "x",
    :ylabel => "y",
    :xticks => get_heatmap_ticks(Nx, Lx),
    :yticks => get_heatmap_ticks(Ny, Ly),
    :xlims => get_heatmap_lims(Nx),
    :ylims => get_heatmap_lims(Ny),
    :dpi => 600,
    :legend => true,
)