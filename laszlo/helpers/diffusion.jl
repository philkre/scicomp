using Distributed
@everywhere using LoopVectorization
using SpecialFunctions


"""
    c_next(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)

Compute the next time step for the diffusion equation using explicit finite differences.

# Arguments
- `c::Matrix{Float64}`: Current concentration field
- `D::Float64`: Diffusion coefficient
- `dx::Float64`: Spatial step size
- `dt::Float64`: Time step size

# Returns
- `Matrix{Float64}`: Updated concentration field

# Boundary Conditions
- Bottom boundary (y=0): c = 0.0
- Top boundary (y=1): c = 1.0
- x-direction: Periodic
"""
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


"""
    c_next_single_loop(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)

Compute the next time step using a single vectorized loop with periodic boundary conditions.

# Arguments
- `c::Matrix{Float64}`: Current concentration field
- `D::Float64`: Diffusion coefficient
- `dx::Float64`: Spatial step size
- `dt::Float64`: Time step size

# Returns
- `Matrix{Float64}`: Updated concentration field

# Notes
Uses a single loop with explicit boundary wrapping for improved performance.
"""
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


"""
    c_next_dist_turbo(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)

Compute the next time step using distributed computing with turbo optimization.

# Arguments
- `c::Matrix{Float64}`: Current concentration field
- `D::Float64`: Diffusion coefficient
- `dx::Float64`: Spatial step size
- `dt::Float64`: Time step size

# Returns
- `Matrix{Float64}`: Updated concentration field

# Notes
Combines `@distributed` and `@turbo` for parallel computation.
"""
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


"""
    c_next_dist(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)

Compute the next time step using distributed computing with flattened indexing.

# Arguments
- `c::Matrix{Float64}`: Current concentration field
- `D::Float64`: Diffusion coefficient
- `dx::Float64`: Spatial step size
- `dt::Float64`: Time step size

# Returns
- `Matrix{Float64}`: Updated concentration field

# Notes
Distributes work by flattening the 2D grid into a 1D range for parallel processing.
"""
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


"""
    c_next_jacobi(c::Matrix{Float64})

Compute the next iteration using the Jacobi method for steady-state diffusion.

# Arguments
- `c::Matrix{Float64}`: Current concentration field

# Returns
- `Matrix{Float64}`: Updated concentration field

# Notes
Uses Fourier number Fo = 0.25 for optimal convergence.
"""
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


"""
    c_next_gauss_seidel!(c::AbstractMatrix{Float64})

Compute the next iteration using the Gauss-Seidel method (in-place).

# Arguments
- `c::AbstractMatrix{Float64}`: Concentration field (modified in-place)

# Returns
- `AbstractMatrix{Float64}`: Reference to the updated concentration field

# Notes
Updates the matrix in-place using the most recently computed values.
"""
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


"""
    c_next_SOR!(c::AbstractMatrix{Float64}, omega::Float64=1.85)

Compute the next iteration using the Successive Over-Relaxation (SOR) method.

# Arguments
- `c::AbstractMatrix{Float64}`: Concentration field (modified in-place)
- `omega::Float64`: Relaxation parameter (default: 1.85)

# Returns
- `AbstractMatrix{Float64}`: Reference to the updated concentration field

# Notes
Omega values: ω = 1 gives Gauss-Seidel, ω > 1 gives over-relaxation, ω < 1 gives under-relaxation.
"""
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


"""
    propagate_c_diffusion(c::Matrix, L::Float64, N::Int, D::Float64, t_0::Float64, t_f::Float64, dt::Float64)

Propagate the diffusion equation from initial time to final time.

# Arguments
- `c::Matrix`: Initial concentration field
- `L::Float64`: Domain length
- `N::Int`: Number of grid points
- `D::Float64`: Diffusion coefficient
- `t_0::Float64`: Initial time
- `t_f::Float64`: Final time
- `dt::Float64`: Time step size

# Returns
- `Matrix`: Concentration field at final time

# Throws
- `ErrorException`: If stability condition D*dt/dx² > 0.25 is violated
"""
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


"""
    c_anal(x::Float64, t::Float64, D::Float64; i_max::Int=100)
using JLD2

Analytical solution for the diffusion equation with boundary conditions c(y=0)=0 and c(y=1)=1.

# Arguments
- `x::Float64`: Position (0 ≤ x ≤ 1)
- `t::Float64`: Time
- `D::Float64`: Diffusion coefficient
- `i_max::Int`: Maximum number of terms in the series expansion (default: 100)

# Returns
- `Float64`: Analytical concentration value at position x and time t

# Notes
Uses a series expansion with complementary error functions. For large `i_max` (> 50,000),
distributes the computation across available workers.
"""
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


"""
    delta(c_old::Matrix{Float64}, c_new::Matrix{Float64})

Compute the maximum absolute difference between two concentration fields.

# Arguments
- `c_old::Matrix{Float64}`: Previous concentration field
- `c_new::Matrix{Float64}`: New concentration field

# Returns
- `Float64`: Maximum absolute difference (L∞ norm)
"""
function delta(c_old::Matrix{Float64}, c_new::Matrix{Float64})
    return maximum(abs.(c_new .- c_old))
end


"""
    stopping_condition(c_old::Matrix{Float64}, c_new::Matrix{Float64}, tol::Float64)

Check if the iterative solver has converged within tolerance.

# Arguments
- `c_old::Matrix{Float64}`: Previous concentration field
- `c_new::Matrix{Float64}`: New concentration field
- `tol::Float64`: Convergence tolerance

# Returns
- `Bool`: True if converged (maximum difference < tolerance)
"""
function stopping_condition(c_old::Matrix{Float64}, c_new::Matrix{Float64}, tol::Float64)
    return delta(c_old, c_new) < tol
end


"""
    solve_until_tol(solver::Function, c_initial::Matrix{Float64}, tol::Float64, max_iters::Int, kwargs...; quiet::Bool=false)

Solve the diffusion equation iteratively until convergence or maximum iterations.

# Arguments
- `solver::Function`: Solver function to use (e.g., c_next_jacobi, c_next_SOR!)
- `c_initial::Matrix{Float64}`: Initial concentration field
- `tol::Float64`: Convergence tolerance
- `max_iters::Int`: Maximum number of iterations
- `kwargs...`: Additional arguments to pass to the solver
- `quiet::Bool`: If true, suppress convergence messages (default: false)

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}}`: Final concentration field and convergence history
"""
function solve_until_tol(solver::Function, c_initial::Matrix{Float64}, tol::Float64, max_iters::Int, args_solver...; quiet::Bool=false, kwargs_solver...)
    # print("Solving with $solver... \n")
    # print(kwargs_solver...)

    c_old = copy(c_initial)
    c_new = copy(c_initial)

    deltas = Float64[]

    for iter in 1:max_iters
        c_new = solver(c_new, args_solver...; kwargs_solver...)
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


"""
    get_iteration_count_SOR(c_0::Matrix{Float64}, omega::Float64, tol::Float64; i_max=20000, check_interval::Int64=1)

Compute the number of iterations required for SOR to converge.

# Arguments
- `c_0::Matrix{Float64}`: Initial concentration field
- `omega::Float64`: Relaxation parameter
- `tol::Float64`: Convergence tolerance
- `i_max::Int`: Maximum iterations (default: 20000)
- `check_interval::Int64`: Check convergence every N iterations (default: 1)

# Returns
- `Int`: Number of iterations required for convergence
"""
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


"""
    c_next_SOR_sink!(c::AbstractMatrix{Float64}, omega::Float64, sink_mask::AbstractMatrix{Bool})

Compute the next iteration using SOR with sink regions where concentration is fixed at zero.

# Arguments
- `c::AbstractMatrix{Float64}`: Concentration field (modified in-place)
- `omega::Float64`: Relaxation parameter
- `sink_mask::AbstractMatrix{Bool}`: Boolean mask indicating sink regions (true = sink)

# Returns
- `AbstractMatrix{Float64}`: Reference to the updated concentration field

# Notes
Cells marked as sinks have their concentration fixed at 0.0 and are not updated by the SOR iteration.
"""
function c_next_SOR_sink!(c::AbstractMatrix{Float64}, omega::Float64, sink_mask::AbstractMatrix{Bool})
    N = size(c, 1)
    Fo = 0.25 * omega

    do_sink = any(sink_mask)

    # # Apply sink mask
    if do_sink
        c[sink_mask] .= 0.0
    end

    @inbounds for i in 1:N
        @inbounds for j in 2:N-1
            # If this cell is a sink, set concentration to 0 and skip update
            if do_sink && sink_mask[i, j]
                continue
            end

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


"""
    c_next_SOR_sink_insulate!(c::AbstractMatrix{Float64}, omega::Float64; sink_mask::AbstractMatrix{Bool}, insulate_mask::AbstractMatrix{Bool})

Compute the next iteration using SOR with both sink regions and insulated boundaries.

# Arguments
- `c::AbstractMatrix{Float64}`: Concentration field (modified in-place)
- `omega::Float64`: Relaxation parameter
- `sink_mask::AbstractMatrix{Bool}`: Boolean mask for sink regions (default: no sinks)
- `insulate_mask::AbstractMatrix{Bool}`: Boolean mask for insulated boundaries (default: no insulation)

# Returns
- `AbstractMatrix{Float64}`: Reference to the updated concentration field

# Notes
- Sink cells: concentration fixed at 0.0, not updated
- Insulated cells: act as reflective boundaries (zero-flux condition)
"""
function c_next_SOR_sink_insulate!(c::AbstractMatrix{Float64}, omega::Float64; sink_mask::AbstractMatrix{Bool}=zeros(Bool, size(c)), insulate_mask::AbstractMatrix{Bool}=zeros(Bool, size(c)))
    N = size(c, 1)
    Fo = 0.25 * omega

    do_sink = any(sink_mask)
    do_insulate = any(insulate_mask)

    # # Apply sink mask
    if do_sink
        c[sink_mask] .= 0.0
    end

    function fetch(requested::Int64, requested_j::Int64; local_i::Int64, local_j::Int64)::Float64
        if do_insulate && insulate_mask[requested, requested_j]
            return c[local_i, local_j]  # Reflect back the value of the local cell (insulated boundary)
        else
            return c[requested, requested_j]
        end
    end

    @inbounds for i in 1:N
        @inbounds for j in 2:N-1
            # If this cell is a sink, set concentration to 0 and skip update
            if do_sink && sink_mask[i, j]
                continue
            end

            i_right = (i == N) ? 1 : i + 1
            i_left = (i == 1) ? N : i - 1

            c[i, j] = Fo * (
                fetch(i_right, j; local_i=i, local_j=j) +
                fetch(i_left, j; local_i=i, local_j=j) +
                fetch(i, j + 1; local_i=i, local_j=j) +
                fetch(i, j - 1; local_i=i, local_j=j)
            ) + (1 - omega) * c[i, j]
        end
    end

    return c
end
