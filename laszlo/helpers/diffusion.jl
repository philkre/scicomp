using Distributed
@everywhere using LoopVectorization
using SpecialFunctions


function c_next(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)
    N = size(c, 1)
    c_new = similar(c)

    # Apply boundary conditions
    c_new[:, 1] .= 0.0  # Bottom boundary
    c_new[:, end] .= 1.0  # Top boundary

    Fo = D * dt / dx^2

    @inbounds @turbo for i in 2:N-1, j in 2:N-1
        c_new[i, j] = c[i, j] + Fo * (c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j])
    end

    # Periodic in x-direction
    @inbounds for j in 2:N-1
        c_new[1, j] = c[1, j] + Fo * (c[2, j] + c[N, j] + c[1, j+1] + c[1, j-1] - 4 * c[1, j])
        c_new[N, j] = c[N, j] + Fo * (c[1, j] + c[N-1, j] + c[N, j+1] + c[N, j-1] - 4 * c[N, j])
    end

    return c_new
end


function c_next_single_loop(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)
    N = size(c, 1)
    c_new = zeros(N, N)

    Fo = (D * dt / dx^2)
    @inbounds @turbo for i in 1:N, j in 2:N-1
        # Project x-end to x-start (periodic boundary condition)
        i_right = i == N ? 2 : i + 1
        i_left = i == 1 ? N - 1 : i - 1

        c_new[i, j] = c[i, j] + Fo * (c[i_right, j] + c[i_left, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j])
    end

    # Apply boundary conditions
    c_new[:, 1] .= 0.0  # Left boundary
    c_new[:, end] .= 1.0  # Right boundary
    return c_new
end


function c_next_dist_turbo(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)
    N = size(c, 1)
    c_new = similar(c)

    # Apply boundary conditions
    c_new[:, 1] .= 0.0  # Left boundary
    c_new[:, end] .= 1.0  # Right boundary

    Fo = D * dt / dx^2

    @inbounds @distributed for i in 2:N-1
        @inbounds @turbo for j in 2:N-1
            c_new[i, j] = c[i, j] + Fo * (c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j])
        end
    end

    @inbounds @turbo for j in 2:N-1
        c_new[1, j] = c[1, j] + Fo * (c[2, j] + c[N, j] + c[1, j+1] + c[1, j-1] - 4 * c[1, j])
        c_new[N, j] = c[N, j] + Fo * (c[1, j] + c[N-1, j] + c[N, j+1] + c[N, j-1] - 4 * c[N, j])
    end

    return c_new
end


function c_next_dist(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)
    N = size(c, 1)
    c_new = similar(c)

    # Apply boundary conditions
    c_new[:, 1] .= 0.0  # Bottom boundary
    c_new[:, end] .= 1.0  # Top boundary

    Fo = D * dt / dx^2

    @inbounds @distributed for z in 1:(N-2)*(N-2)
        i = mod(div(z - 1, N - 2), N - 2) + 2
        j = mod(z - 1, N - 2) + 2
        c_new[i, j] = c[i, j] + Fo * (c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4 * c[i, j])
    end

    @inbounds @turbo for j in 2:N-1
        c_new[1, j] = c[1, j] + Fo * (c[2, j] + c[N, j] + c[1, j+1] + c[1, j-1] - 4 * c[1, j])
        c_new[N, j] = c[N, j] + Fo * (c[1, j] + c[N-1, j] + c[N, j+1] + c[N, j-1] - 4 * c[N, j])
    end

    return c_new
end


function c_next_jacobi(c::Matrix{Float64})
    N = size(c, 1)
    c_new = similar(c)

    # Apply boundary conditions
    c_new[:, 1] .= 0.0  # Bottom boundary
    c_new[:, end] .= 1.0  # Top boundary

    Fo = 0.25

    @inbounds @turbo for i in 2:N-1, j in 2:N-1
        c_new[i, j] = Fo * (c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1])
    end

    # Periodic in x-direction
    @inbounds for j in 2:N-1
        c_new[1, j] = Fo * (c[2, j] + c[N, j] + c[1, j+1] + c[1, j-1])
        c_new[N, j] = Fo * (c[1, j] + c[N-1, j] + c[N, j+1] + c[N, j-1])
    end

    return c_new
end


function c_next_gauss_seidel!(c::AbstractMatrix{Float64})
    N = size(c, 1)
    Fo = 0.25

    @inbounds for i in 1:N
        @inbounds for j in 2:N-1
            i_left = i == 1 ? N : i - 1
            i_right = i == N ? 1 : i + 1
            c[i, j] = Fo * (c[i_right, j] + c[i_left, j] + c[i, j+1] + c[i, j-1])
        end
    end

    return c
end


function c_next_SOR!(c::AbstractMatrix{Float64}, omega::Float64=1.85)
    N = size(c, 1)
    Fo = 0.25 * omega

    @inbounds for i in 1:N
        @inbounds for j in 2:N-1
            i_right = (i == N) ? 1 : i + 1
            i_left = (i == 1) ? N : i - 1

            c[i, j] = Fo * (
                c[i_right, j] +
                c[i_left, j] +
                c[i, j+1] +
                c[i, j-1]
            ) + (1 - omega) * c[i, j]
        end
    end

    return c
end


function propagate_c_diffusion(c::Matrix, L::Float64, N::Int, D::Float64, t_0::Float64, t_f::Float64, dt::Float64)
    c_curr = copy(c)
    dx = L / N

    # Check stability condition
    if D * dt / dx^2 > 0.25
        error("Stability condition violated: D*dt/dx^2 must be <= 0.25")
    end

    for t in t_0:dt:t_f
        c_curr = c_next(c_curr, D, dx, dt)
    end

    return c_curr
end


"Analytical solution for the diffusion equation with c(y=0)=0 and c(y=1)=1."
function c_anal(x::Float64, t::Float64, D::Float64; i_max::Int=100)
    denom = 2 * sqrt(D * t)

    function c_anal_i(x::Float64, i::Int)
        return erfc((1 - x + 2i) / denom) - erfc((1 + x + 2i) / denom)
    end

    if i_max > 50_000
        return @distributed (+) for i in 0:i_max
            c_anal_i(x, i)
        end
    else
        return sum(c_anal_i(x, i) for i in 0:i_max)
    end
end


function delta(c_old::Matrix{Float64}, c_new::Matrix{Float64})
    return maximum(abs.(c_new .- c_old))
end


function stopping_condition(c_old::Matrix{Float64}, c_new::Matrix{Float64}, tol::Float64)
    return delta(c_old, c_new) < tol
end


function solve_until_tol(solver::Function, c_initial::Matrix{Float64}, tol::Float64, max_iters::Int, kwargs...; quiet::Bool=false)
    c_old = copy(c_initial)
    c_new = copy(c_initial)

    deltas = Float64[]
    for iter in 1:max_iters
        c_new = solver(c_new, kwargs...)
        push!(deltas, delta(c_old, c_new))

        if stopping_condition(c_old, c_new, tol)
            if !quiet
                println("$solver converged after $iter iterations ")
            end
            break
        end

        copyto!(c_old, c_new)
    end

    return c_new, deltas
end


function get_iteration_count_SOR(c_0::Matrix{Float64}, omega::Float64, tol::Float64; i_max=20000, check_interval::Int64=1)
    c_new = copy(c_0)

    for i in 1:i_max
        if i % check_interval == 0
            c_old = copy(c_new)
            c_next_SOR!(c_new, omega)
            if stopping_condition(c_old, c_new, tol)
                return i
            end
        else
            c_next_SOR!(c_new, omega)
        end
    end

    return i
end