module DLACore

using Metal: MtlMatrix, PrivateStorage
using Distributions: Categorical

# Diffusion helpers
include("diffusion.jl")
using .Diffusion: solve_until_tol, c_next_SOR_sink_red_black!, solve_until_tol_metal!, c_next_SOR_sink_metal!


FloatMatrix = Union{Matrix{Float64},Matrix{Float32}}


function choose_candidate(cpu_c::FloatMatrix, cpu_sink::Matrix{Bool}, eta::Float64)
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

function diffusion_limited_aggregation_step!(
    c::Union{Matrix{Float64},MtlMatrix{Float32,PrivateStorage}},
    c_sink::Union{Matrix{Bool},MtlMatrix{Bool,PrivateStorage}},
    c_source::Matrix{Bool},
    eta::Float64,
    cpu_c::FloatMatrix,
    cpu_sink::Matrix{Bool}
    ;
    tol::Float64=1e-6,
    i_max_conv::Int=10_000,
    omega_sor::Float64=1.9,
    use_GPU::Bool=false,
    c_old::Union{MtlMatrix{Float32,PrivateStorage},Nothing}=nothing,
    diffs::Union{MtlMatrix{Float32,PrivateStorage},Nothing}=nothing)
    if use_GPU
        copyto!(cpu_c, solve_until_tol_metal!(c_next_SOR_sink_metal!, c, tol, i_max_conv, Float32(omega_sor), c_sink; check_every=25, c_old=c_old, diffs=diffs, track_deltas=false, quiet=true))
    else
        c = solve_until_tol(c_next_SOR_sink_red_black!, c, tol, i_max_conv, omega_sor, c_sink; track_deltas=false, quiet=true)
        copyto!(cpu_c, c)
    end

    chosen_cell = choose_candidate(cpu_c, cpu_sink, eta)

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