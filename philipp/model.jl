module Model

using SpecialFunctions
using StatsBase: Weights, sample

export wave1d!, diffusion2d!, laplace_jacobi!, analytical_profile_series, euler_step!, leapfrog_step!, laplace_gauss_seidel!, laplace_sor!, laplace_red_black_sor!, laplace_multigrid!, update_mask!

"""
    wave1d!(psi, a, c, dx)

Compute the spatial second-derivative term of the 1D wave equation
`d^2 psi / dt^2 = c^2 d^2 psi / dx^2` using central differences and store the result in `a`.
Boundary accelerations are fixed to zero.
"""
function wave1d!(psi::AbstractVector{<:Real}, a::AbstractVector{<:Real}, c::Real, dx::Real)
    a[1] = 0
    a[end] = 0

    @inbounds for i in 2:length(psi)-1
        a[i] = c^2 * (psi[i+1] - 2 * psi[i] + psi[i-1]) / dx^2
    end

    return nothing
end

"""
    diffusion2d!(c, D, dx, dt)

Advance the 2D diffusion equation by one explicit Euler step on a uniform grid.
Uses periodic boundaries in `x` and Dirichlet boundaries in `y` (`c[:,1]=0`, `c[:,end]=1`).
"""
function diffusion2d!(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)
    c_curr = copy(c)
    max_i, max_j = size(c)

    @inbounds for j in 2:max_j-1
        for i in 1:max_i
            ip = (i == max_i) ? 1 : i + 1
            im = (i == 1) ? max_i : i - 1

            c[i, j] = c_curr[i, j] + D * dt / (dx * dx) * (
                c_curr[ip, j] + c_curr[im, j] +
                c_curr[i, j+1] + c_curr[i, j-1] -
                4 * c_curr[i, j]
            )
        end
    end

    # Dirichlet in y
    c[:, 1] .= 0
    c[:, end] .= 1

    return nothing
end

"""
    _apply_sink!(c, sink_indices)

Set sink cells to zero concentration in-place.
No-op when `sink_indices === nothing`.
"""
function _apply_sink!(
    c::Matrix{Float64},
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}},
)
    if isnothing(sink_indices)
        return nothing
    end
    c[sink_indices] .= 0.0
    return nothing
end

"""
    laplace_jacobi!(c; sink_indices=nothing)

Perform one Jacobi iteration for the 2D Laplace equation on `c`.
Uses periodic boundaries in `x` and Dirichlet boundaries in `y`.
Optional `sink_indices` are clamped to zero each sweep.
"""
function laplace_jacobi!(
    c::Matrix{Float64};
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}}=nothing,
)
    c_curr = copy(c)
    max_i, max_j = size(c)

    @inbounds for j in 2:max_j-1
        for i in 1:max_i
            ip = (i == max_i) ? 1 : i + 1
            im = (i == 1) ? max_i : i - 1

            c[i, j] = 0.25 * (
                c_curr[ip, j] + c_curr[im, j] +
                c_curr[i, j+1] + c_curr[i, j-1]
            )
            _apply_sink!(c, sink_indices)
        end
    end

    # Dirichlet in y
    c[:, 1] .= 0
    c[:, end] .= 1

    return nothing
end

"""
    laplace_gauss_seidel!(c; sink_indices=nothing)

Perform one Gauss-Seidel iteration for the 2D Laplace equation on `c`.
Uses periodic boundaries in `x` and Dirichlet boundaries in `y`.
Optional `sink_indices` are clamped to zero each sweep.
"""
function laplace_gauss_seidel!(
    c::Matrix{Float64};
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}}=nothing,
)
    max_i, max_j = size(c)

    @inbounds for j in 2:max_j-1
        for i in 1:max_i
            ip = (i == max_i) ? 1 : i + 1
            im = (i == 1) ? max_i : i - 1

            c[i, j] = 0.25 * (
                c[ip, j] + c[im, j] +
                c[i, j+1] + c[i, j-1]
            )
            _apply_sink!(c, sink_indices)
        end
    end

    # Dirichlet in y
    c[:, 1] .= 0
    c[:, end] .= 1

    # Periodic in x
    c[1, :] .= c[end, :]

    return nothing
end

"""
    laplace_sor!(c, omega; sink_indices=nothing)

Perform one SOR iteration for the 2D Laplace equation with relaxation factor `omega`.
Uses periodic boundaries in `x` and Dirichlet boundaries in `y`.
Optional `sink_indices` are clamped to zero each sweep.
Returns the maximum absolute per-cell update in the sweep.
"""
function laplace_sor!(
    c::Matrix{Float64},
    omega::Float64;
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}}=nothing,
)
    max_i, max_j = size(c)
    max_delta = 0.0

    @inbounds for j in 2:max_j-1
        for i in 1:max_i
            ip = (i == max_i) ? 1 : i + 1
            im = (i == 1) ? max_i : i - 1

            c_new = 0.25 * (
                c[ip, j] + c[im, j] +
                c[i, j+1] + c[i, j-1]
            )
            c_old = c[i, j]
            c[i, j] = (1 - omega) * c_old + omega * c_new
            delta = abs(c[i, j] - c_old)
            if delta > max_delta
                max_delta = delta
            end
        end
    end

    _apply_sink!(c, sink_indices)

    # Dirichlet in y
    c[:, 1] .= 0
    c[:, end] .= 1

    return max_delta
end

"""
    laplace_red_black_sor!(c, omega; sink_indices=nothing)

Perform one red-black SOR sweep for the 2D Laplace equation.
Uses periodic boundaries in `x` and Dirichlet boundaries in `y`.
Optional `sink_indices` are clamped to zero each sweep.
Returns the maximum absolute per-cell update in the sweep.
"""
function laplace_red_black_sor!(
    c::Matrix{Float64},
    omega::Float64;
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}}=nothing,
)
    max_i, max_j = size(c)
    max_delta = 0.0

    @inbounds for color in 0:1
        for j in 2:max_j-1
            i_start = if color == 0
                isodd(j) ? 1 : 2
            else
                isodd(j) ? 2 : 1
            end
            for i in i_start:2:max_i
                ip = (i == max_i) ? 1 : i + 1
                im = (i == 1) ? max_i : i - 1

                c_new = 0.25 * (
                    c[ip, j] + c[im, j] +
                    c[i, j+1] + c[i, j-1]
                )
                c_old = c[i, j]
                c[i, j] = (1 - omega) * c_old + omega * c_new
                delta = abs(c[i, j] - c_old)
                if delta > max_delta
                    max_delta = delta
                end
            end
        end
    end

    _apply_sink!(c, sink_indices)

    # Dirichlet in y
    c[:, 1] .= 0
    c[:, end] .= 1

    clamp!(c, 0.0, Inf)

    return max_delta
end

"""
    _sink_mask_from_indices(dims, sink_indices)

Construct a boolean sink mask of size `dims` from sink index vectors.
Returns an all-false mask when `sink_indices === nothing`.
"""
function _sink_mask_from_indices(
    dims::Tuple{Int,Int},
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}},
)
    mask = falses(dims...)
    if !isnothing(sink_indices)
        mask[sink_indices] .= true
    end
    return mask
end

"""
    _coarsen_mask(mask)

Coarsen a fine-grid sink mask by 2x2 blocking.
A coarse cell is marked sink if any contributing fine cell is sink.
Dirichlet boundary rows in `y` are forced to non-sink.
"""
function _coarsen_mask(mask::BitMatrix)
    ni, nj = size(mask)
    ci = max(2, cld(ni, 2))
    cj = max(2, cld(nj, 2))
    coarse = falses(ci, cj)

    @inbounds for jc in 1:cj
        j1 = 2 * jc - 1
        j2 = min(2 * jc, nj)
        @simd for ic in 1:ci
            i1 = 2 * ic - 1
            i2 = min(2 * ic, ni)
            v = mask[i1, j1]
            if i2 != i1
                v |= mask[i2, j1]
            end
            if j2 != j1
                v |= mask[i1, j2]
                if i2 != i1
                    v |= mask[i2, j2]
                end
            end
            coarse[ic, jc] = v
        end
    end

    coarse[:, 1] .= false
    coarse[:, end] .= false
    return coarse
end

"""
    _restrict_avg(a)

Restrict a fine-grid scalar field to a coarser grid using block averaging.
Supports odd grid sizes by averaging partial edge blocks.
"""
function _restrict_avg(a::Matrix{Float64})
    ni, nj = size(a)
    ci = max(2, cld(ni, 2))
    cj = max(2, cld(nj, 2))
    coarse = zeros(ci, cj)

    @inbounds for jc in 1:cj
        j1 = 2 * jc - 1
        j2 = min(2 * jc, nj)
        @simd for ic in 1:ci
            i1 = 2 * ic - 1
            i2 = min(2 * ic, ni)
            s = a[i1, j1]
            n = 1
            if i2 != i1
                s += a[i2, j1]
                n += 1
            end
            if j2 != j1
                s += a[i1, j2]
                n += 1
                if i2 != i1
                    s += a[i2, j2]
                    n += 1
                end
            end
            coarse[ic, jc] = s / n
        end
    end

    return coarse
end

"""
    _prolong_bilinear!(fine, coarse)

Bilinearly interpolate a coarse-grid field onto a fine grid in-place.
The output `fine` is overwritten.
"""
function _prolong_bilinear!(fine::Matrix{Float64}, coarse::Matrix{Float64})
    ni, nj = size(fine)
    ci, cj = size(coarse)
    inv_ni = 1.0 / max(1, ni - 1)
    inv_nj = 1.0 / max(1, nj - 1)
    sx = ci - 1
    sy = cj - 1

    i0v = Vector{Int}(undef, ni)
    i1v = Vector{Int}(undef, ni)
    txv = Vector{Float64}(undef, ni)
    @inbounds @simd for i in 1:ni
        x = (i - 1) * sx * inv_ni + 1.0
        i0 = clamp(floor(Int, x), 1, ci)
        i1 = min(i0 + 1, ci)
        i0v[i] = i0
        i1v[i] = i1
        txv[i] = x - i0
    end

    @inbounds for j in 1:nj
        y = (j - 1) * sy * inv_nj + 1.0
        j0 = clamp(floor(Int, y), 1, cj)
        j1 = min(j0 + 1, cj)
        ty = y - j0
        @simd for i in 1:ni
            i0 = i0v[i]
            i1 = i1v[i]
            tx = txv[i]

            c00 = coarse[i0, j0]
            c10 = coarse[i1, j0]
            c01 = coarse[i0, j1]
            c11 = coarse[i1, j1]

            fine[i, j] = (1 - tx) * (1 - ty) * c00 +
                         tx * (1 - ty) * c10 +
                         (1 - tx) * ty * c01 +
                         tx * ty * c11
        end
    end

    return nothing
end

function _mg_smooth_step!(
    u::Matrix{Float64},
    omega::Float64,
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}},
    smoother::Symbol,
)
    if smoother == :sor
        return laplace_sor!(u, omega; sink_indices=sink_indices)
    elseif smoother == :rb_sor
        return laplace_red_black_sor!(u, omega; sink_indices=sink_indices)
    else
        throw(ArgumentError("Unknown multigrid smoother '$smoother'. Use :sor or :rb_sor."))
    end
end

"""
    _laplace_mg_vcycle!(u, sink_mask, level, max_level, omega, pre_sweeps, post_sweeps, coarse_sweeps)

Perform one recursive multigrid V-cycle on `u` with sink constraints.
Applies pre/post SOR smoothing, coarse-grid correction, boundary enforcement,
and returns the maximum update magnitude observed in smoothing.
"""
function _laplace_mg_vcycle!(
    u::Matrix{Float64},
    sink_mask::BitMatrix,
    level::Int,
    max_level::Int,
    omega::Float64,
    smoother::Symbol,
    pre_sweeps::Int,
    post_sweeps::Int,
    coarse_sweeps::Int,
)
    sink_indices = findall(sink_mask)
    if level >= max_level || min(size(u)...) <= 5
        max_delta = 0.0
        for _ in 1:coarse_sweeps
            max_delta = max(max_delta, _mg_smooth_step!(u, omega, sink_indices, smoother))
        end
        return max_delta
    end

    max_delta = 0.0
    for _ in 1:pre_sweeps
        max_delta = max(max_delta, _mg_smooth_step!(u, omega, sink_indices, smoother))
    end

    u_coarse = _restrict_avg(u)
    sink_mask_coarse = _coarsen_mask(sink_mask)
    _laplace_mg_vcycle!(
        u_coarse,
        sink_mask_coarse,
        level + 1,
        max_level,
        omega,
        smoother,
        pre_sweeps,
        post_sweeps,
        coarse_sweeps,
    )

    _prolong_bilinear!(u, u_coarse)
    _apply_sink!(u, sink_indices)
    u[:, 1] .= 0.0
    u[:, end] .= 1.0
    clamp!(u, 0.0, Inf)

    for _ in 1:post_sweeps
        max_delta = max(max_delta, _mg_smooth_step!(u, omega, sink_indices, smoother))
    end

    return max_delta
end

"""
    laplace_multigrid!(c; sink_indices=nothing, omega=1.6, smoother=:sor, ncycles=8, levels=0, pre_sweeps=2, post_sweeps=2, coarse_sweeps=30, tol=1e-6)

Run V-cycle multigrid iterations for the 2D Laplace equation on `c`.
Uses periodic boundaries in `x` and Dirichlet boundaries in `y`.
The smoothing scheme can be selected with `smoother` (`:sor` or `:rb_sor`).
Returns the maximum absolute update from the final cycle.
"""
function laplace_multigrid!(
    c::Matrix{Float64};
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}}=nothing,
    omega::Float64=1.6,
    smoother::Symbol=:sor,
    ncycles::Int=8,
    levels::Int=0,
    pre_sweeps::Int=2,
    post_sweeps::Int=2,
    coarse_sweeps::Int=30,
    tol::Float64=1e-6,
)
    if ncycles < 1
        throw(ArgumentError("ncycles must be >= 1"))
    end

    ni, nj = size(c)
    max_levels_auto = 1
    while min(ni, nj) >= 9
        ni = cld(ni, 2)
        nj = cld(nj, 2)
        max_levels_auto += 1
    end
    nlevels = levels > 0 ? min(levels, max_levels_auto) : max_levels_auto

    sink_mask = _sink_mask_from_indices(size(c), sink_indices)
    max_delta = Inf
    for _ in 1:ncycles
        max_delta = _laplace_mg_vcycle!(
            c,
            sink_mask,
            1,
            nlevels,
            omega,
            smoother,
            pre_sweeps,
            post_sweeps,
            coarse_sweeps,
        )
        if max_delta < tol
            break
        end
    end
    return max_delta
end

"""
    update_mask!(c, mask, nu)

Advance the DLA aggregate by one growth event.
Find free sites adjacent to occupied sites, compute growth weights `max(c, 0)^nu`,
sample one candidate, and mark it occupied in `mask` (set to `0.0`).
Uses periodic wrapping in `x` and bounded neighbors in `y`.
"""
function update_mask!(c, mask, nu)
    max_i, max_j = size(mask)

    mask_candidates = Tuple{Int,Int}[]
    candidate_weights = Float64[]
    candidate_seen = falses(size(mask))

    for I in findall(mask .== 0.0)
        i, j = Tuple(I)
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            nj = j + dj
            if nj < 1 || nj > max_j
                continue
            end

            ni = i + di
            if ni < 1
                ni = max_i
            elseif ni > max_i
                ni = 1
            end

            if mask[ni, nj] == 1.0 && !candidate_seen[ni, nj]
                push!(mask_candidates, (ni, nj))
                candidate_seen[ni, nj] = true
                push!(candidate_weights, max(c[ni, nj], 0.0)^nu)
            end
        end
    end

    if !isempty(mask_candidates)
        wsum = sum(candidate_weights)
        chosen_index = if isfinite(wsum) && wsum > 0.0
            sample(1:length(mask_candidates), Weights(candidate_weights))
        else
            rand(1:length(mask_candidates))
        end
        chosen_site = mask_candidates[chosen_index]
        mask[chosen_site...] = 0.0
    end

    return nothing
end

"""
    analytical_profile_series(x, t, D, L; n_terms=200)

Compute a truncated image-series solution for the 1D diffusion profile with
Dirichlet boundaries on `[0, L]` at time `t`.
"""
function analytical_profile_series(
    x::AbstractVector{<:Real},
    t::Real,
    D::Real,
    L::Real;
    n_terms::Integer=200,
)
    if t <= 0
        throw(ArgumentError("t must be > 0 for analytical_profile_series"))
    end

    denom = 2 * sqrt(D * t)
    c = zeros(Float64, length(x))

    for i in 0:n_terms
        c .+= erfc.((L .- x .+ 2i * L) ./ denom) .- erfc.((L .+ x .+ 2i * L) ./ denom)
    end

    return c
end

"""
    euler_step!(psi, a, v, c, dx, dt)

Advance the 1D wave state by one explicit Euler step in-place.
"""
function euler_step!(
    psi::AbstractVector{<:Real},
    a::AbstractVector{<:Real},
    v::AbstractVector{<:Real},
    c::Real,
    dx::Real,
    dt::Real,
)
    wave1d!(psi, a, c, dx)

    @inbounds for i in 2:length(psi)-1
        v[i] += a[i] * dt
        psi[i] += v[i] * dt
    end

    psi[1] = 0
    psi[end] = 0
    v[1] = 0
    v[end] = 0

    return nothing
end

"""
    leapfrog_step!(psi, a, v, c, dx, dt)

Advance the 1D wave state by one leapfrog step in-place using
a half-kick, drift, half-kick update.
"""
function leapfrog_step!(
    psi::AbstractVector{<:Real},
    a::AbstractVector{<:Real},
    v::AbstractVector{<:Real},
    c::Real,
    dx::Real,
    dt::Real,
)
    wave1d!(psi, a, c, dx)

    @inbounds for i in 2:length(psi)-1
        v[i] += 0.5 * dt * a[i]
        psi[i] += v[i] * dt
    end

    psi[1] = 0
    psi[end] = 0

    wave1d!(psi, a, c, dx)

    @inbounds for i in 2:length(psi)-1
        v[i] += 0.5 * dt * a[i]
    end

    v[1] = 0
    v[end] = 0

    return nothing
end

end
