module DLACore

using Metal: MtlMatrix, PrivateStorage
using Distributions: Categorical

# Diffusion helpers
include("diffusion.jl")
using .Diffusion: solve_until_tol, c_next_SOR_sink_red_black!, solve_until_tol_metal!, c_next_SOR_sink_metal!


FloatMatrix = Union{Matrix{Float64},Matrix{Float32}}


"""
    choose_candidate(cpu_c::FloatMatrix, cpu_sink::Matrix{Bool}; eta::Float64, kwargs_dump...)::CartesianIndex{2}

Choose a candidate cell for aggregation based on diffusion probabilities.

# Arguments
- `cpu_c::FloatMatrix`: Current concentration field on CPU
- `cpu_sink::Matrix{Bool}`: Boolean mask indicating sink regions (already aggregated)
- `eta::Float64`: Exponent for probability weighting (controls how much concentration influences selection)
- `kwargs_dump...`: Additional keyword arguments (ignored)

# Returns
- `CartesianIndex{2}`: Coordinates of the chosen candidate cell

# Notes
- Only considers cells neighboring existing sink cells
- Selection probability is proportional to c^eta where c is the concentration
- Higher eta values favor cells with higher concentration
"""
function choose_candidate(cpu_c::FloatMatrix, cpu_sink::Matrix{Bool}; eta::Float64, kwargs_dump...)::CartesianIndex{2}
    neighbor_mask = falses(size(cpu_sink))
    # Copy source stamp to neighbor candidates
    neighbor_mask[2:end, :] .|= cpu_sink[1:end-1, :]    # below source
    neighbor_mask[1:end-1, :] .|= cpu_sink[2:end, :]    # above source
    neighbor_mask[:, 2:end] .|= cpu_sink[:, 1:end-1]    # right of source
    neighbor_mask[:, 1:end-1] .|= cpu_sink[:, 2:end]    # left of source
    # Exclude original source cells from neighbors
    neighbor_mask .&= .!cpu_sink

    candidates = findall(neighbor_mask)
    # diffusion probabilities are proportional to concentration at candidate cells
    # Raise to the power eta, control floating point issues by maxing with 0.0
    probabilities = max.(cpu_c[candidates], 0.0) .^ eta
    # Normalize
    probabilities ./= sum(probabilities)
    # Choose candidate cell based on probabilities
    chosen_index = rand(Categorical(probabilities))
    chosen_cell = candidates[chosen_index]

    return chosen_cell
end


"""
    get_neighboring_cells(pos::CartesianIndex{2}; loop_x::Bool=true, Ni::Int=nothing)::Vector{CartesianIndex{2}}

Get the four neighboring cells (up, down, left, right) of a given position.

# Arguments
- `pos::CartesianIndex{2}`: Current position
- `loop_x::Bool`: If true, apply periodic boundary conditions in x-direction (default: true)
- `Ni::Int`: Grid size in x-direction (required if loop_x=true)

# Returns
- `Vector{CartesianIndex{2}}`: Vector of 4 neighboring cell coordinates

# Notes
- Returns neighbors in order: right, left, above, below
- With loop_x=true, wraps around x-boundaries (periodic boundary conditions)
- y-direction does not wrap (absorbing boundaries)
"""
function get_neighboring_cells(pos::CartesianIndex{2}; loop_x::Bool=true, Ni::Int=nothing)::Vector{CartesianIndex{2},}
    candidates = [
        CartesianIndex(pos[1] + 1, pos[2]),  # right
        CartesianIndex(pos[1] - 1, pos[2]),  # left
        CartesianIndex(pos[1], pos[2] + 1),  # above
        CartesianIndex(pos[1], pos[2] - 1)   # below
    ]

    if loop_x
        # Handle x-boundaries by teleporting to opposite side
        if candidates[1][1] > Ni
            candidates[1] = CartesianIndex(candidates[1][1] - Ni, candidates[1][2])
        end
        if candidates[2][1] < 1
            candidates[2] = CartesianIndex(Ni + candidates[2][1], candidates[2][2])
        end
    end

    return candidates
end


"""
    choose_candidate_monte_carlo(cpu_c::FloatMatrix, cpu_sink::Matrix{Bool}; p_s::Float64=1.0, i_max::Int=1_000_000, c_source::Matrix{Bool}, kwargs_dump...)::CartesianIndex{2}

Choose a candidate cell for aggregation using Monte Carlo random walk simulation.

# Arguments
- `cpu_c::FloatMatrix`: Current concentration field on CPU (not used in this method)
- `cpu_sink::Matrix{Bool}`: Boolean mask indicating sink regions (already aggregated)
- `p_s::Float64`: Sticking probability when neighboring a sink (default: 1.0)
- `i_max::Int`: Maximum number of random walk steps (default: 1,000,000)
- `c_source::Matrix{Bool}`: Boolean mask indicating source regions (starting points)
- `kwargs_dump...`: Additional keyword arguments (ignored)

# Returns
- `CartesianIndex{2}`: Coordinates of the chosen candidate cell

# Throws
- `ErrorException`: If no candidate is found within i_max iterations

# Notes
- Performs random walk starting from a random source cell
- When walk reaches a cell neighboring a sink, it sticks with probability p_s
- Periodic boundary conditions in x-direction, absorbing boundaries in y-direction
- More physically realistic than concentration-based selection for certain phenomena
"""
function choose_candidate_monte_carlo(
    cpu_c::FloatMatrix,
    cpu_sink::Matrix{Bool},
    ;
    p_s::Float64=1.0,
    i_max::Int=1_000_000,
    c_source::Matrix{Bool},
    kwargs_dump...)::CartesianIndex{2}

    # Start random walk from a random source cell, here we just take the first one for simplicity
    pos::CartesianIndex{2} = rand(findall(c_source))

    Ni = size(cpu_c, 1)
    Nj = size(cpu_c, 2)

    for i in 1:i_max
        # Check y-boundaries
        if pos[2] < 1 || pos[2] > Nj
            # Restart
            pos = rand(findall(c_source))
            continue
        end

        # Double check if we are on a sink cell, if so, restart (can happen due to teleporting)
        if cpu_sink[pos]
            pos = rand(findall(c_source))
            continue
        end

        neighbors = get_neighboring_cells(pos; loop_x=true, Ni=Ni)
        # Filter neighbors that are inside the grid in y-direction
        neighbors_inside = filter((p) -> p[2] >= 1 && p[2] <= Nj, neighbors)
        # Find which of the neighboring cells are sinks
        neighboring_sinks = findall(cpu_sink[neighbors_inside])

        # Check if position borders a sink cell, if so, return it with probability p_s
        if !isempty(neighboring_sinks) && rand() < p_s
            return pos
        end

        # Find all neighbors that are not sinks, allow teleporting in y-direction 
        neighbors_not_sinks = filter((p) -> p[2] < 1 || p[2] > Nj || !cpu_sink[p], neighbors)

        # Move to a random neighboring, not allowed to move onto a sink cell, but can teleport in x-direction, will restart if teleporting in y-direction
        pos = rand(neighbors_not_sinks)
    end

    error("Didn't find candidate cell in $i_max iterations, consider increasing i_max or check if there are any valid candidates at all.")
end


"""
    diffusion_limited_aggregation_step!(c, c_sink, c_source, cpu_c, cpu_sink; kwargs...)

Perform one step of diffusion-limited aggregation (DLA) simulation.

# Arguments
- `c::Union{Matrix{Float64},MtlMatrix{Float32,PrivateStorage}}`: Concentration field (GPU or CPU)
- `c_sink::Union{Matrix{Bool},MtlMatrix{Bool,PrivateStorage}}`: Sink mask (modified in-place)
- `c_source::Matrix{Bool}`: Source mask (constant boundary)
- `cpu_c::FloatMatrix`: CPU copy of concentration field
- `cpu_sink::Matrix{Bool}`: CPU copy of sink mask

# Keyword Arguments
- `tol::Float64`: Convergence tolerance for SOR solver (default: 1e-6)
- `i_max_conv::Int`: Maximum iterations for SOR solver (default: 10,000)
- `omega_sor::Float64`: Relaxation parameter for SOR (default: 1.9)
- `eta::Union{Float64,Nothing}`: Exponent for concentration-based candidate selection
- `p_s::Union{Float64,Nothing}`: Sticking probability for Monte Carlo candidate selection
- `candidate_picker::Function`: Function to select next aggregation site (default: choose_candidate)
- `use_GPU::Bool`: Use Metal GPU acceleration if available (default: false)
- `c_old::Union{MtlMatrix{Float32,PrivateStorage},Nothing}`: Preallocated array for GPU solver
- `diffs::Union{MtlMatrix{Float32,PrivateStorage},Nothing}`: Preallocated array for GPU solver

# Notes
1. Solves steady-state diffusion equation with current sink configuration using SOR
2. Selects candidate cell for aggregation using specified picker function
3. Adds selected cell to sink if it's not a source cell
4. Modifies `c_sink` and `cpu_sink` in-place

# Algorithm
This implements one iteration of the DLA process:
- Solve Laplace equation: ∇²c = 0 with Dirichlet boundaries
- Select growth site based on concentration field or random walk
- Aggregate selected cell into the growing structure
"""
function diffusion_limited_aggregation_step!(
    c::Union{Matrix{Float64},MtlMatrix{Float32,PrivateStorage}},
    c_sink::Union{Matrix{Bool},MtlMatrix{Bool,PrivateStorage}},
    c_source::Matrix{Bool},
    cpu_c::FloatMatrix,
    cpu_sink::Matrix{Bool}
    ;
    tol::Float64=1e-6,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.9,
    eta::Union{Float64,Nothing}=nothing,
    p_s::Union{Float64,Nothing}=nothing,
    candidate_picker::Function=choose_candidate,
    use_GPU::Bool=false,
    c_old::Union{MtlMatrix{Float32,PrivateStorage},Nothing}=nothing,
    diffs::Union{MtlMatrix{Float32,PrivateStorage},Nothing}=nothing)
    if use_GPU
        copyto!(cpu_c, solve_until_tol_metal!(c_next_SOR_sink_metal!, c, tol, i_max_conv, Float32(omega_sor), c_sink; check_every=25, c_old=c_old, diffs=diffs, track_deltas=false, quiet=true))
    else
        c = solve_until_tol(c_next_SOR_sink_red_black!, c, tol, i_max_conv, omega_sor, c_sink; track_deltas=false, quiet=true)
        copyto!(cpu_c, c)
    end
    chosen_cell::CartesianIndex{2} = candidate_picker(cpu_c, cpu_sink; eta=eta, p_s=p_s, c_source=c_source)

    # Not allowed to change source into sink
    if c_source[chosen_cell]
        return
    end

    # Strike the lightning: set chosen cell as new source
    cpu_sink[chosen_cell] = true
    copyto!(c_sink, cpu_sink)

    return
end


end # module