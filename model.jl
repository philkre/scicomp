module Model

using SpecialFunctions

export wave1d!, diffusion2d!, laplace_jacobi!, analytical_profile_series, euler_step!, leapfrog_step!, laplace_gauss_seidel!, laplace_sor!

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
            ip = (i == max_i) ? 2 : i + 1
            im = (i == 1) ? max_i - 1 : i - 1

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

    # Periodic in x
    c[1, :] .= c[end, :]

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
            ip = (i == max_i) ? 2 : i + 1
            im = (i == 1) ? max_i - 1 : i - 1

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

    # Periodic in x
    c[1, :] .= c[end, :]

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
            ip = (i == max_i) ? 2 : i + 1
            im = (i == 1) ? max_i - 1 : i - 1

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
"""
function laplace_sor!(
    c::Matrix{Float64},
    omega::Float64;
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}}=nothing,
)
    max_i, max_j = size(c)

    @inbounds for j in 2:max_j-1
        for i in 1:max_i
            ip = (i == max_i) ? 2 : i + 1
            im = (i == 1) ? max_i - 1 : i - 1

            c_new = 0.25 * (
                c[ip, j] + c[im, j] +
                c[i, j+1] + c[i, j-1]
            )
            c[i, j] = (1 - omega) * c[i, j] + omega * c_new
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
