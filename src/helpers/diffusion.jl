module Diffusion

using Distributed
using LoopVectorization: @turbo
using SpecialFunctions: erfc
using Metal: MtlMatrix, MtlDeviceMatrix, @metal, PrivateStorage, thread_position_in_grid_2d
using CUDA


"""
    laplacian_2d(c::Matrix{Float64}, dx::Float64; periodic_x::Bool=true, periodic_y::Bool=true)::Matrix{Float64}

Compute the 2D Laplacian ∇²c using finite differences.

# Arguments
- `c::Matrix{Float64}`: Concentration field
- `dx::Float64`: Grid spacing (assumed equal in x and y)
- `periodic_x::Bool`: Use periodic boundary conditions in x-direction (default: true)
- `periodic_y::Bool`: Use periodic boundary conditions in y-direction (default: true)

# Returns
- `Matrix{Float64}`: Laplacian of the concentration field

# Notes
Uses 5-point stencil: ∇²c ≈ (c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1] - 4c[i,j]) / dx²
"""
function laplacian_2d(c::Matrix{Float64}, dx::Float64; periodic_x::Bool=true, periodic_y::Bool=true)::Matrix{Float64}
    N, M = size(c)
    laplacian = zeros(N, M)

    Fo = 1.0 / dx^2

    # Do everything except boundaries
    @inbounds @turbo for i in 2:N-1, j in 2:M-1
        # 5-point stencil for Laplacian
        laplacian[i, j] = Fo * (
            c[i+1, j] +
            c[i-1, j] +
            c[i, j+1] +
            c[i, j-1]
        ) - 4 * Fo * c[i, j]
    end

    # Handle x boundaries
    if periodic_x
        @inbounds @turbo for j in 2:M-1
            # Left boundary
            laplacian[1, j] = Fo * (
                c[2, j] +
                c[N, j] +
                c[1, j+1] +
                c[1, j-1]
            ) - 4 * Fo * c[1, j]
            # Right boundary
            laplacian[N, j] = Fo * (
                c[1, j] +
                c[N-1, j] +
                c[N, j+1] +
                c[N, j-1]
            ) - 4 * Fo * c[N, j]
        end
    else
        @inbounds @turbo for j in 2:M-1
            # Left boundary
            laplacian[1, j] = Fo * (
                c[2, j] +
                c[1, j] +
                c[1, j+1] +
                c[1, j-1]
            ) - 4 * Fo * c[1, j]
            # Right boundary
            laplacian[N, j] = Fo * (
                c[N, j] +
                c[N-1, j] +
                c[N, j+1] +
                c[N, j-1]
            ) - 4 * Fo * c[N, j]
        end
    end

    # Handle y boundaries
    if periodic_y
        @inbounds @turbo for i in 2:N-1
            # Bottom boundary
            laplacian[i, 1] = Fo * (
                c[i+1, 1] +
                c[i-1, 1] +
                c[i, 2] +
                c[i, M]
            ) - 4 * Fo * c[i, 1]
            # Top boundary
            laplacian[i, M] = Fo * (
                c[i+1, M] +
                c[i-1, M] +
                c[i, 1] +
                c[i, M-1]
            ) - 4 * Fo * c[i, M]
        end
    else
        @inbounds @turbo for i in 2:N-1
            # Bottom boundary
            laplacian[i, 1] = Fo * (
                c[i+1, 1] +
                c[i-1, 1] +
                c[i, 2] +
                c[i, 1]
            ) - 4 * Fo * c[i, 1]
            # Top boundary
            laplacian[i, M] = Fo * (
                c[i+1, M] +
                c[i-1, M] +
                c[i, M] +
                c[i, M-1]
            ) - 4 * Fo * c[i, M]
        end
    end

    return laplacian
end


"""
    c_next(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)::Matrix{Float64}

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
function c_next(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)::Matrix{Float64}
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
    c_next_single_loop(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)::Matrix{Float64}

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
function c_next_single_loop(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)::Matrix{Float64}
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
    c_next_dist_turbo(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)::Matrix{Float64}

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
function c_next_dist_turbo(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)::Matrix{Float64}
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
    c_next_dist(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)::Matrix{Float64}

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
function c_next_dist(c::Matrix{Float64}, D::Float64, dx::Float64, dt::Float64)::Matrix{Float64}
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
    c_next_jacobi(c::Matrix{Float64})::Matrix{Float64}

Compute the next iteration using the Jacobi method for steady-state diffusion.

# Arguments
- `c::Matrix{Float64}`: Current concentration field

# Returns
- `Matrix{Float64}`: Updated concentration field

# Notes
Uses Fourier number Fo = 0.25 for optimal convergence.
"""
function c_next_jacobi(c::Matrix{Float64})::Matrix{Float64}
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
    c_next_gauss_seidel!(c::Matrix{Float64})::Matrix{Float64}

Compute the next iteration using the Gauss-Seidel method (in-place).

# Arguments
- `c::Matrix{Float64}`: Concentration field (modified in-place)

# Returns
- `Matrix{Float64}`: Reference to the updated concentration field

# Notes
Updates the matrix in-place using the most recently computed values.
"""
function c_next_gauss_seidel!(c::Matrix{Float64})::Matrix{Float64}
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
    c_next_SOR!(c::Matrix{Float64}, omega::Float64=1.85)::Matrix{Float64}

Compute the next iteration using the Successive Over-Relaxation (SOR) method.

# Arguments
- `c::Matrix{Float64}`: Concentration field (modified in-place)
- `omega::Float64`: Relaxation parameter (default: 1.85)

# Returns
- `Matrix{Float64}`: Reference to the updated concentration field

# Notes
Omega values: ω = 1 gives Gauss-Seidel, ω > 1 gives over-relaxation, ω < 1 gives under-relaxation.
"""
function c_next_SOR!(c::Matrix{Float64}, omega::Float64=1.85)::Matrix{Float64}
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
    propagate_c_diffusion(c::Matrix, L::Float64, N::Int, D::Float64, t_0::Float64, t_f::Float64, dt::Float64)::Matrix{Float64}

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
function propagate_c_diffusion(c::Matrix, L::Float64, N::Int, D::Float64, t_0::Float64, t_f::Float64, dt::Float64)::Matrix{Float64}
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
    c_anal(x::Float64, t::Float64, D::Float64; i_max::Int=100)::Float64
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
function c_anal(x::Float64, t::Float64, D::Float64; i_max::Int=100)::Float64
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
    analytical_sol(t::Float64, D::Float64, L::Float64, N::Int64)

Lambda function to compute the analytical solution for the diffusion equation over a discrete grid.

# Arguments
- `t::Float64`: Time
- `D::Float64`: Diffusion coefficient
- `L::Float64`: Domain length
- `N::Int64`: Number of grid points

# Returns
- `Vector{Float64}`: Analytical concentration values at grid points
"""
analytical_sol = (t::Float64, D::Float64, L::Float64, N::Int64) -> [c_anal(x, t, D) for x in LinRange(0, L, N)]


"""
    c_anal_2d(N::Int)::Matrix{Float64}

Generate the analytical steady-state solution for 2D diffusion with linear gradient in y-direction.

# Arguments
- `N::Int`: Grid size (N×N)

# Returns
- `Matrix{Float64}`: Concentration field with linear gradient from 0 at y=0 to 1 at y=1

# Notes
This represents the steady-state solution where c(y=0)=0 and c(y=1)=1 with periodic boundaries in x.
"""
function c_anal_2d(N::Int)::Matrix{Float64}
    c_0 = zeros(N, N)
    c_y = range(0, stop=1, length=N)
    c_0 .= c_y'  # Broadcast row vector to all rows
    return c_0
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
function stopping_condition(c_old::Matrix{Float64}, c_new::Matrix{Float64}, tol::Float64)::Bool
    return delta(c_old, c_new) < tol
end


"""
    solve_until_tol(solver::Function, c_initial::Matrix{Float64}, tol::Float64, max_iters::Int, kwargs...; quiet::Bool=false)::Tuple{Matrix{Float64},Vector{Float64}}

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
function solve_until_tol(solver::Function, c_initial::Matrix{Float64}, tol::Float64, max_iters::Int, args_solver...; quiet::Bool=false, track_deltas::Bool=true, kwargs_solver...)::Union{Tuple{Matrix{Float64},Vector{Float64}},Matrix{Float64}}
    c_old = copy(c_initial)
    c_new = copy(c_initial)

    if track_deltas
        deltas = Float64[]
    end

    for iter in 1:max_iters
        copyto!(c_new, solver(c_new, args_solver...; kwargs_solver...))

        delta_curr = delta(c_old, c_new)
        if track_deltas
            push!(deltas, delta_curr)
        end

        if delta_curr < tol
            if !quiet
                println("$solver converged after $iter iterations ")
            end
            if track_deltas
                return c_new, deltas
            else
                return c_new
            end
        end

        copyto!(c_old, c_new)
    end

    if !quiet
        println("$solver did not converge after $max_iters iterations")
    end

    if track_deltas
        return c_new, deltas
    else
        return c_new
    end
end


"""
    delta_metal!(diffs::MtlMatrix{Float32,PrivateStorage}, c_old::MtlMatrix{Float32,PrivateStorage}, c_new::MtlMatrix{Float32,PrivateStorage})

Compute the maximum absolute difference between two Metal GPU arrays.

# Arguments
- `diffs::MtlMatrix{Float32,PrivateStorage}`: Preallocated array to store element-wise differences
- `c_old::MtlMatrix{Float32,PrivateStorage}`: Previous concentration field
- `c_new::MtlMatrix{Float32,PrivateStorage}`: New concentration field

# Returns
- `Float32`: Maximum absolute difference (L∞ norm)

# Notes
Computes differences on GPU and only syncs the maximum scalar value to CPU for efficiency.
"""
function delta_metal!(diffs::MtlMatrix{Float32,PrivateStorage}, c_old::MtlMatrix{Float32,PrivateStorage}, c_new::MtlMatrix{Float32,PrivateStorage})
    N = size(c_old, 1)

    # 2D configuration
    threads_per_group = (16, 16)  # 1024 threads total per group
    groups = (cld(N, 16), cld(N - 2, 16))  # Cover all rows and N-2 columns
    @metal threads = threads_per_group groups = groups abs_diff_kernel_metal!(N, c_old, c_new, diffs)

    return maximum(diffs)  # still syncs, but now we call it rarely
end


"""
    abs_diff_kernel_metal!(N::Int, c_old::MtlDeviceMatrix{Float32,1}, c_new::MtlDeviceMatrix{Float32,1}, diffs::MtlDeviceMatrix{Float32,1})

Metal GPU kernel to compute element-wise absolute differences between concentration fields.

# Arguments
- `N::Int`: Grid size
- `c_old::MtlDeviceMatrix{Float32,1}`: Previous concentration field on device
- `c_new::MtlDeviceMatrix{Float32,1}`: New concentration field on device
- `diffs::MtlDeviceMatrix{Float32,1}`: Output array for differences on device

# Notes
Skips boundary cells (first and last row) since they have fixed boundary conditions.
"""
function abs_diff_kernel_metal!(N::Int, c_old::MtlDeviceMatrix{Float32,1}, c_new::MtlDeviceMatrix{Float32,1}, diffs::MtlDeviceMatrix{Float32,1})
    (i, j) = thread_position_in_grid_2d()

    if i > N || j > N - 2 || i == 1 || i == N
        return
    end

    diffs[i, j] = abs(c_new[i, j] - c_old[i, j])

    return
end


"""
    solve_until_tol_metal!(solver!::Function, c::MtlMatrix{Float32,PrivateStorage}, tol::Float64, i_max::Int, args_solver...; quiet::Bool=false, track_deltas::Bool=false, check_every::Int=100, c_old::MtlMatrix{Float32,PrivateStorage}=similar(c), diffs::MtlMatrix{Float32,PrivateStorage}=similar(c), kwargs_solver...)

Solve iteratively on Metal GPU until convergence or maximum iterations.

# Arguments
- `solver!::Function`: Metal solver function to use
- `c::MtlMatrix{Float32,PrivateStorage}`: Initial concentration field (modified in-place)
- `tol::Float64`: Convergence tolerance
- `i_max::Int`: Maximum number of iterations
- `args_solver...`: Additional arguments to pass to the solver
- `quiet::Bool`: If true, suppress convergence messages (default: false)
- `track_deltas::Bool`: If true, track convergence history (default: false)
- `check_every::Int`: Check convergence every N iterations (default: 100)
- `c_old::MtlMatrix{Float32,PrivateStorage}`: Preallocated array for old values
- `diffs::MtlMatrix{Float32,PrivateStorage}`: Preallocated array for differences
- `kwargs_solver...`: Additional keyword arguments for the solver

# Returns
- `MtlMatrix{Float32,PrivateStorage}` or `Tuple{MtlMatrix{Float32,PrivateStorage}, Vector{Float32}}`: Final concentration field and optionally convergence history

# Notes
Checks convergence periodically to minimize GPU-CPU synchronization overhead.
"""
function solve_until_tol_metal!(
    solver!::Function,
    c::MtlMatrix{Float32,PrivateStorage},
    tol::Float64,
    i_max::Int,
    args_solver...
    ;
    quiet::Bool=false,
    track_deltas::Bool=false,
    check_every::Int=100,
    c_old::MtlMatrix{Float32,PrivateStorage}=similar(c),
    diffs::MtlMatrix{Float32,PrivateStorage}=similar(c),  # allocate once
    kwargs_solver...)::Union{MtlMatrix{Float32,PrivateStorage},Tuple{MtlMatrix{Float32,PrivateStorage},Vector{Float32}}}
    if track_deltas
        deltas = Float32[]
    end

    for i in 1:i_max
        # Only sync every `check_every` iterations to copy c to c_old and compute delta, otherwise let GPU run asynchronously
        if i % check_every == 0
            copyto!(c_old, c)
        end

        # Update
        solver!(c, args_solver...; kwargs_solver...)

        # Check convergence every `check_every` iterations
        if i % check_every == 0
            # Compute delta on GPU and only sync to CPU for the single scalar value, instead of syncing entire c matrix every iteration
            delta = delta_metal!(diffs, c_old, c)
            if track_deltas
                push!(deltas, delta)
            end

            # Stopping condition
            if delta < tol
                if !quiet
                    println("$solver! converged after $i iterations ")
                end

                if track_deltas
                    return c, deltas
                else
                    return c
                end
            end
        end
    end

    if !quiet
        println("$solver! did not converge after $i_max iterations")
    end

    if track_deltas
        return c, deltas
    else
        return c
    end
end


"""
    solve_until_tol_cuda!(solver!::Function, c::CuArray{Float32,2}, tol::Float64, i_max::Int, args_solver...; quiet::Bool=false, track_deltas::Bool=false, check_every::Int=100, c_old::CuArray{Float32,2}=similar(c), diffs::CuArray{Float32,2}=similar(c), kwargs_solver...)

Solve iteratively on CUDA GPU until convergence or maximum iterations.

# Arguments
- `solver!::Function`: CUDA solver function to use
- `c::CuArray{Float32,2}`: Initial concentration field (modified in-place)
- `tol::Float64`: Convergence tolerance
- `i_max::Int`: Maximum number of iterations
- `args_solver...`: Additional arguments to pass to the solver
- `quiet::Bool`: If true, suppress convergence messages (default: false)
- `track_deltas::Bool`: If true, track convergence history (default: false)
- `check_every::Int`: Check convergence every N iterations (default: 100)
- `c_old::CuArray{Float32,2}`: Preallocated array for old values
- `diffs::CuArray{Float32,2}`: Preallocated array for differences
- `kwargs_solver...`: Additional keyword arguments for the solver

# Returns
- `CuArray{Float32,2}` or `Tuple{CuArray{Float32,2}, Vector{Float32}}`: Final concentration field and optionally convergence history
"""
function solve_until_tol_cuda!(
    solver!::Function,
    c::CuArray{Float32,2},
    tol::Float64,
    i_max::Int,
    args_solver...
    ;
    quiet::Bool=false,
    track_deltas::Bool=false,
    check_every::Int=100,
    c_old::CuArray{Float32,2}=similar(c),
    diffs::CuArray{Float32,2}=similar(c),  # allocate once
    kwargs_solver...)::Union{CuArray{Float32,2},Tuple{CuArray{Float32,2},Vector{Float32}}}
    if track_deltas
        deltas = Float32[]
    end

    for i in 1:i_max
        # Only sync every `check_every` iterations to copy c to c_old and compute delta, otherwise let GPU run asynchronously
        if i % check_every == 0
            copyto!(c_old, c)
        end

        # Update
        solver!(c, args_solver...; kwargs_solver...)

        # Check convergence every `check_every` iterations
        if i % check_every == 0
            # Compute delta on GPU and only sync to CPU for the single scalar value, instead of syncing entire c matrix every iteration
            delta = delta_cuda!(diffs, c_old, c)
            if track_deltas
                push!(deltas, delta)
            end

            # Stopping condition
            if delta < tol
                if !quiet
                    println("$solver! converged after $i iterations ")
                end

                if track_deltas
                    return c, deltas
                else
                    return c
                end
            end
        end
    end

    if !quiet
        println("$solver! did not converge after $i_max iterations")
    end

    if track_deltas
        return c, deltas
    else
        return c
    end
end


"""
    delta_cuda!(diffs::CuArray{Float32,2}, c_old::CuArray{Float32,2}, c_new::CuArray{Float32,2})

Compute the maximum absolute difference between two CUDA arrays.

# Arguments
- `diffs::CuArray{Float32,2}`: Preallocated array to store differences
- `c_old::CuArray{Float32,2}`: Previous concentration field
- `c_new::CuArray{Float32,2}`: New concentration field

# Returns
- `Float32`: Maximum absolute difference
"""
function delta_cuda!(diffs::CuArray{Float32,2}, c_old::CuArray{Float32,2}, c_new::CuArray{Float32,2})
    N = size(c_old, 1)

    # 2D configuration
    threads_per_block = (16, 16)  # 256 threads total per block
    blocks = (cld(N, 16), cld(N - 2, 16))  # Cover all rows and N-2 columns
    @cuda threads = threads_per_block blocks = blocks abs_diff_kernel_cuda!(N, c_old, c_new, diffs)

    return maximum(diffs)  # still syncs, but now we call it rarely
end


"""
    max_abs_diff_kernel_cuda!(c_old::CuDeviceArray{Float32,2}, c_new::CuDeviceArray{Float32,2}, diffs::CuDeviceArray{Float32,2})

CUDA kernel to compute element-wise absolute differences.

# Arguments
- `c_old::CuDeviceArray{Float32,2}`: Previous concentration field
- `c_new::CuDeviceArray{Float32,2}`: New concentration field
- `diffs::CuDeviceArray{Float32,2}`: Output array for differences
"""
function abs_diff_kernel_cuda!(N::Int, c_old::CuDeviceArray{Float32,2}, c_new::CuDeviceArray{Float32,2}, diffs::CuDeviceArray{Float32,2})
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > N || j > N - 2 || i == 1 || i == N
        return
    end

    if i <= size(c_old, 1) && j <= size(c_old, 2)
        diffs[i, j] = abs(c_new[i, j] - c_old[i, j])
    end

    return
end


"""
    get_iteration_count_SOR(c_0::Matrix{Float64}, omega::Float64, tol::Float64; i_max=20000, check_interval::Int64=1)::Int64

Compute the number of iterations required for SOR to converge.

# Arguments
- `c_0::Matrix{Float64}`: Initial concentration field
- `omega::Float64`: Relaxation parameter
- `tol::Float64`: Convergence tolerance
- `i_max::Int`: Maximum iterations (default: 20000)
- `check_interval::Int64`: Check convergence every N iterations (default: 1)

# Returns
- `Int64`: Number of iterations required for convergence
"""
function get_iteration_count_SOR(c_0::Matrix{Float64}, omega::Float64, tol::Float64; i_max=60_000, check_interval::Int64=1)::Int64
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

    @warn "SOR did not converge within $i_max iterations for omega = $omega"

    return i_max
end


"""
    c_next_SOR_sink!(c::Matrix{Float64}, omega::Float64, sink_mask::Matrix{Bool})::Matrix{Float64}

Compute the next iteration using SOR with sink regions where concentration is fixed at zero.

# Arguments
- `c::Matrix{Float64}`: Concentration field (modified in-place)
- `omega::Float64`: Relaxation parameter
- `sink_mask::Matrix{Bool}`: Boolean mask indicating sink regions (true = sink)

# Returns
- `Matrix{Float64}`: Reference to the updated concentration field

# Notes
Cells marked as sinks have their concentration fixed at 0.0 and are not updated by the SOR iteration.
"""
function c_next_SOR_sink!(c::Matrix{Float64}, omega::Float64, sink_mask::Matrix{Bool})::Matrix{Float64}
    N = size(c, 1)
    Fo = 0.25 * omega

    do_sink = any(sink_mask)

    # Apply sink mask
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
    c_next_SOR_sink_red_black!(c::Matrix{Float64}, omega::Float64, sink_mask::Matrix{Bool})::Matrix{Float64}

Compute the next iteration using SOR with red-black ordering and sink regions.

# Arguments
- `c::Matrix{Float64}`: Concentration field (modified in-place)
- `omega::Float64`: Relaxation parameter
- `sink_mask::Matrix{Bool}`: Boolean mask indicating sink regions (true = sink)

# Returns
- `Matrix{Float64}`: Reference to the updated concentration field

# Notes
Uses red-black Gauss-Seidel ordering to enable vectorization with `@turbo`. Updates are split into:
- Red cells: where (i+j) is even
- Black cells: where (i+j) is odd

Cells marked as sinks have their concentration fixed at 0.0 after both passes complete.
"""
function c_next_SOR_sink_red_black!(c::Matrix{Float64}, omega::Float64, sink_mask::Matrix{Bool})::Matrix{Float64}
    N = size(c, 1)
    Fo = 0.25 * omega

    # Parity-based red-black sweep valid for both even and odd N.
    @inbounds for color in 0:1
        for j in 2:N-1
            i_start = if color == 0
                isodd(j) ? 1 : 2
            else
                isodd(j) ? 2 : 1
            end

            for i in i_start:2:N
                if sink_mask[i, j]
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
        # Keep sink cells pinned between color sweeps.
        c[sink_mask] .= 0.0
    end

    return c
end


"""
    c_next_SOR_sink_red_black!(c::Matrix{Float64}, omega::Float64, sink_mask::Matrix{Bool})::Matrix{Float64}

Compute the next iteration using SOR with red-black ordering and sink regions.

# Arguments
- `c::Matrix{Float64}`: Concentration field (modified in-place)
- `omega::Float64`: Relaxation parameter
- `sink_mask::Matrix{Bool}`: Boolean mask indicating sink regions (true = sink)

# Returns
- `Matrix{Float64}`: Reference to the updated concentration field

# Notes
Uses red-black Gauss-Seidel ordering to enable vectorization with `@turbo`. Updates are split into:
- Red cells: where (i+j) is even
- Black cells: where (i+j) is odd

Cells marked as sinks have their concentration fixed at 0.0 after both passes complete.
"""
function c_next_SOR_sink_red_black_turbo!(c::Matrix{Float64}, omega::Float64, sink_mask::Matrix{Bool})::Matrix{Float64}
    N = size(c, 1)
    if isodd(N)
        @warn "Current implementation of c_next_SOR_sink_red_black! assumes an even grid size for proper red-black ordering. Results may be incorrect for odd N."
    end
    Fo = 0.25 * omega

    # Red pass: i+j = even 
    # Red pass: update cells for even rows, even columns (even + even = even)
    @inbounds for i in 2:2:N-1
        @inbounds @turbo for j in 2:2:N-1
            c[i, j] = Fo * (
                c[i+1, j] +
                c[i-1, j] +
                c[i, j+1] +
                c[i, j-1]
            ) + (1 - omega) * c[i, j]
        end
    end
    # Red pass: update cells for uneven rows, uneven columns (odd + odd = even)
    @inbounds for i in 3:2:N-1
        @inbounds @turbo for j in 3:2:N-1
            c[i, j] = Fo * (
                c[i+1, j] +
                c[i-1, j] +
                c[i, j+1] +
                c[i, j-1]
            ) + (1 - omega) * c[i, j]
        end
    end
    # Do boundary pass - Red cells (i+j = even) on the boundaries
    @inbounds @turbo for j in 3:2:N-1
        # Left boundary
        c[1, j] = Fo * (
            c[2, j] +
            c[N, j] +
            c[1, j+1] +
            c[1, j-1]
        ) + (1 - omega) * c[1, j]

    end
    @inbounds @turbo for j in 2:2:N-1
        # Right boundary
        c[N, j] = Fo * (
            c[1, j] +
            c[N-1, j] +
            c[N, j+1] +
            c[N, j-1]
        ) + (1 - omega) * c[N, j]
    end

    # Apply sink mask after red pass to ensure sinks remain at 0 before black pass updates
    c[sink_mask] .= 0.0

    # Black pass: i+j = odd
    # Black pass: update cells for even rows, uneven columns (even + odd = odd)
    @inbounds for i in 2:2:N-1
        @inbounds @turbo for j in 3:2:N-1
            c[i, j] = Fo * (
                c[i+1, j] +
                c[i-1, j] +
                c[i, j+1] +
                c[i, j-1]
            ) + (1 - omega) * c[i, j]
        end
    end
    # Black pass: update cells for uneven rows, even columns (odd + even = odd)
    @inbounds for i in 3:2:N-1
        @inbounds @turbo for j in 2:2:N-1
            c[i, j] = Fo * (
                c[i+1, j] +
                c[i-1, j] +
                c[i, j+1] +
                c[i, j-1]
            ) + (1 - omega) * c[i, j]
        end
    end
    # Do boundary pass - Black cells (i+j = odd) on the boundaries
    @inbounds @turbo for j in 2:2:N-1
        # Left boundary
        c[1, j] = Fo * (
            c[2, j] +
            c[N, j] +
            c[1, j+1] +
            c[1, j-1]
        ) + (1 - omega) * c[1, j]
    end
    @inbounds @turbo for j in 3:2:N-1
        # Right boundary
        c[N, j] = Fo * (
            c[1, j] +
            c[N-1, j] +
            c[N, j+1] +
            c[N, j-1]
        ) + (1 - omega) * c[N, j]
    end

    # Apply sink mask after both passes
    c[sink_mask] .= 0.0

    return c
end


"""
    c_next_SOR_kernel_metal!(c::MtlDeviceMatrix{Float32,1}, sink_mask::MtlDeviceMatrix{Bool,1}, Fo::Float32, omega::Float32, N::Int, color::Int32)

Metal GPU kernel for SOR iteration with red-black ordering and sink regions.

# Arguments
- `c::MtlDeviceMatrix{Float32,1}`: Concentration field on device
- `sink_mask::MtlDeviceMatrix{Bool,1}`: Boolean mask indicating sink regions on device
- `Fo::Float32`: Fourier number factor (0.25 * omega)
- `omega::Float32`: Relaxation parameter
- `N::Int`: Grid size
- `color::Int32`: Red-black ordering flag (0 for red cells where i+j is even, 1 for black cells where i+j is odd)

# Notes
- Uses red-black ordering to avoid race conditions in parallel updates
- Cells marked as sinks have their concentration fixed at 0.0
- Periodic boundary conditions in x-direction
"""
function c_next_SOR_kernel_metal!(c::MtlDeviceMatrix{Float32,1}, sink_mask::MtlDeviceMatrix{Bool,1}, Fo::Float32, omega::Float32, N::Int, color::Int32)
    i = thread_position_in_grid_2d().x
    j = thread_position_in_grid_2d().y + 1 # Shift j by 1 to account for skipping first and last column

    # Red-black ordering: only update if (i+j) mod 2 matches color
    if ((i + j) & 1) != color
        return
    end

    # Check bound
    if i > N || j >= N
        return
    end

    # Check mask and exit early if this cell is a sink
    if sink_mask[i, j]
        c[i, j] = 0.0f0
        return
    end

    i_right = (i == N) ? 1 : i + 1
    i_left = (i == 1) ? N : i - 1

    c[i, j] = Fo * (
        c[i_right, j] +
        c[i_left, j] +
        c[i, j+1] +
        c[i, j-1]
    ) + (1 - omega) * c[i, j]
    return
end


"""
    c_next_SOR_sink_metal!(c::MtlMatrix{Float32}, omega::Float32, sink_mask::MtlMatrix{Bool})::MtlMatrix{Float32}

Compute the next iteration using SOR on Metal GPU with sink regions.

# Arguments
- `c::MtlMatrix{Float32}`: Concentration field (modified in-place)
- `omega::Float32`: Relaxation parameter
- `sink_mask::MtlMatrix{Bool}`: Boolean mask indicating sink regions (true = sink)

# Returns
- `MtlMatrix{Float32}`: Reference to the updated concentration field

# Notes
- Uses red-black Gauss-Seidel ordering for in-place updates on GPU
- Two kernel launches: one for red cells (i+j even), one for black cells (i+j odd)
- Cells marked as sinks have their concentration fixed at 0.0
- Periodic boundary conditions in x-direction
"""
function c_next_SOR_sink_metal!(c::MtlMatrix{Float32}, omega::Float32, sink_mask::MtlMatrix{Bool})::MtlMatrix{Float32}
    N = size(c, 1)
    Fo = Float32(0.25) * omega

    # 2D configuration
    threads_per_group = (16, 16)  # 256 threads total per group
    groups = (cld(N, 16), cld(N - 2, 16))  # Cover all rows and N-2 columns

    # Red-black ordering for in-place SOR updates
    # Red pass: update cells where (i+j) is even
    @metal threads = threads_per_group groups = groups c_next_SOR_kernel_metal!(c, sink_mask, Fo, omega, N, Int32(0))

    # Black pass: update cells where (i+j) is odd (no sync - let GPU schedule)
    @metal threads = threads_per_group groups = groups c_next_SOR_kernel_metal!(c, sink_mask, Fo, omega, N, Int32(1))

    return c
end



"""
    c_next_SOR_kernel_cuda!(c::CuDeviceArray{Float32,2}, sink_mask::CuDeviceArray{Bool,2}, Fo::Float32, omega::Float32, N::Int, color::Int32)

CUDA kernel for SOR iteration with red-black ordering and sink regions.

# Arguments
- `c::CuDeviceArray{Float32,2}`: Concentration field
- `sink_mask::CuDeviceArray{Bool,2}`: Boolean mask indicating sink regions
- `Fo::Float32`: Fourier number factor
- `omega::Float32`: Relaxation parameter
- `N::Int`: Grid size
- `color::Int32`: Red-black ordering flag (0 for red, 1 for black)
"""
function c_next_SOR_kernel_cuda!(c::CuDeviceArray{Float32,2}, sink_mask::CuDeviceArray{Bool,2}, Fo::Float32, omega::Float32, N::Int, color::Int32)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y + 1  # Shift j by 1 to account for skipping first and last column

    # Red-black ordering: only update if (i+j) mod 2 matches color
    if ((i + j) & 1) != color
        return
    end

    # Check bounds
    if i > N || j >= N
        return
    end

    # Check mask and exit early if this cell is a sink
    if sink_mask[i, j]
        c[i, j] = 0.0f0
        return
    end

    i_right = (i == N) ? 1 : i + 1
    i_left = (i == 1) ? N : i - 1

    c[i, j] = Fo * (
        c[i_right, j] +
        c[i_left, j] +
        c[i, j+1] +
        c[i, j-1]
    ) + (1 - omega) * c[i, j]
    return
end


"""
    c_next_SOR_sink_cuda!(c::CuArray{Float32,2}, omega::Float32, sink_mask::CuArray{Bool,2})::CuArray{Float32,2}

Compute the next iteration using SOR on CUDA GPU with sink regions.

# Arguments
- `c::CuArray{Float32,2}`: Concentration field (modified in-place)
- `omega::Float32`: Relaxation parameter
- `sink_mask::CuArray{Bool,2}`: Boolean mask indicating sink regions (true = sink)

# Returns
- `CuArray{Float32,2}`: Reference to the updated concentration field

# Notes
Uses red-black Gauss-Seidel ordering for in-place updates on GPU. Cells marked as sinks have their concentration fixed at 0.0.
"""
function c_next_SOR_sink_cuda!(c::CuArray{Float32,2}, omega::Float32, sink_mask::CuArray{Bool,2})::CuArray{Float32,2}
    N = size(c, 1)
    Fo = Float32(0.25) * omega

    # 2D configuration
    threads_per_block = (16, 16)  # 256 threads total per block
    blocks = (cld(N, 16), cld(N - 2, 16))  # Cover all rows and N-2 columns

    # Red-black ordering for in-place SOR updates
    # Red pass: update cells where (i+j) is even
    @cuda threads = threads_per_block blocks = blocks c_next_SOR_kernel_cuda!(c, sink_mask, Fo, omega, N, Int32(0))

    # Black pass: update cells where (i+j) is odd (no sync - let GPU schedule)
    @cuda threads = threads_per_block blocks = blocks c_next_SOR_kernel_cuda!(c, sink_mask, Fo, omega, N, Int32(1))

    return c
end


"""
    c_next_SOR_sink_insulate!(c::Matrix{Float64}, omega::Float64; sink_mask::Matrix{Bool}, insulate_mask::Matrix{Bool})::Matrix{Float64}

Compute the next iteration using SOR with both sink regions and insulated boundaries.

# Arguments
- `c::Matrix{Float64}`: Concentration field (modified in-place)
- `omega::Float64`: Relaxation parameter
- `sink_mask::Matrix{Bool}`: Boolean mask for sink regions (default: no sinks)
- `insulate_mask::Matrix{Bool}`: Boolean mask for insulated boundaries (default: no insulation)

# Returns
- `Matrix{Float64}`: Reference to the updated concentration field

# Notes
- Sink cells: concentration fixed at 0.0, not updated
- Insulated cells: act as reflective boundaries (zero-flux condition)
"""
function c_next_SOR_sink_insulate!(c::Matrix{Float64}, omega::Float64; sink_mask::Matrix{Bool}=zeros(Bool, size(c)), insulate_mask::Matrix{Bool}=zeros(Bool, size(c)))::Matrix{Float64}
    N = size(c, 1)
    Fo = 0.25 * omega

    do_sink = any(sink_mask)
    do_insulate = any(insulate_mask)

    # Apply sink mask
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


"""
    _sink_mask_from_indices(dims, sink_indices)

Build a boolean sink mask from index vectors.
"""
function _sink_mask_from_indices(
    dims::Tuple{Int,Int},
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}},
)::Matrix{Bool}
    sink_mask = zeros(Bool, dims...)
    if !isnothing(sink_indices)
        sink_mask[sink_indices] .= true
    end
    return sink_mask
end


"""
    _coarsen_mask(mask)

Coarsen a sink mask by 2x2 blocks (OR-reduction).
"""
function _coarsen_mask(mask::AbstractMatrix{Bool})::Matrix{Bool}
    ni, nj = size(mask)
    ci = max(2, cld(ni, 2))
    cj = max(2, cld(nj, 2))
    coarse = zeros(Bool, ci, cj)
    _coarsen_mask!(coarse, mask)
    return coarse
end

"""
    _coarsen_mask!(coarse, mask)

In-place coarsening of a sink mask by 2x2 blocks (OR-reduction).
"""
function _coarsen_mask!(coarse::Matrix{Bool}, mask::AbstractMatrix{Bool})
    ni, nj = size(mask)
    ci, cj = size(coarse)
    @inbounds for jc in 1:cj
        j1 = 2 * jc - 1
        j2 = min(2 * jc, nj)
        for ic in 1:ci
            i1 = 2 * ic - 1
            i2 = min(2 * ic, ni)
            coarse[ic, jc] = mask[i1, j1] || mask[i2, j1] || mask[i1, j2] || mask[i2, j2]
        end
    end

    coarse[:, 1] .= false
    coarse[:, end] .= false
    return nothing
end


"""
    _restrict_avg(a)

Restrict a fine grid to a coarse grid by local averaging on 2x2 blocks.
"""
function _restrict_avg(a::Matrix{Float64})::Matrix{Float64}
    ni, nj = size(a)
    ci = max(2, cld(ni, 2))
    cj = max(2, cld(nj, 2))
    coarse = zeros(Float64, ci, cj)
    _restrict_avg!(coarse, a)
    return coarse
end

"""
    _restrict_avg!(coarse, a)

In-place restriction from fine grid `a` into preallocated `coarse`
using local 2x2 averaging.
"""
function _restrict_avg!(coarse::Matrix{Float64}, a::Matrix{Float64})
    ni, nj = size(a)
    ci, cj = size(coarse)
    @inbounds for jc in 1:cj
        j1 = 2 * jc - 1
        j2 = min(2 * jc, nj)
        for ic in 1:ci
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

    return nothing
end


"""
    _prolong_bilinear!(fine, coarse)

Bilinearly prolong a coarse grid into `fine` in-place.
"""
function _prolong_bilinear!(fine::Matrix{Float64}, coarse::Matrix{Float64})
    ni, nj = size(fine)
    ci, cj = size(coarse)
    inv_ni = 1.0 / max(1, ni - 1)
    inv_nj = 1.0 / max(1, nj - 1)
    sx = ci - 1
    sy = cj - 1

    @inbounds for j in 1:nj
        y = (j - 1) * sy * inv_nj + 1.0
        j0 = clamp(floor(Int, y), 1, cj)
        j1 = min(j0 + 1, cj)
        ty = y - j0
        for i in 1:ni
            x = (i - 1) * sx * inv_ni + 1.0
            i0 = clamp(floor(Int, x), 1, ci)
            i1 = min(i0 + 1, ci)
            tx = x - i0

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


"""
    _apply_bc_and_sink!(u, sink_mask)

Apply boundary and sink constraints in-place:
- Dirichlet in y (`u[:,1]=0`, `u[:,end]=1`)
- sink cells fixed to `0`
- clamp concentration to non-negative values
"""
function _apply_bc_and_sink!(u::Matrix{Float64}, sink_mask::Matrix{Bool})
    u[:, 1] .= 0.0
    u[:, end] .= 1.0
    u[sink_mask] .= 0.0
    clamp!(u, 0.0, Inf)
    return nothing
end


"""
    _mg_smooth_step_cpu!(u, omega, sink_mask, smoother)

Perform one CPU smoothing sweep used by multigrid.
Supports `smoother = :sor` or `:rb_sor`.
"""
function _mg_smooth_step_cpu!(u::Matrix{Float64}, omega::Float64, sink_mask::Matrix{Bool}, smoother::Symbol)
    if smoother == :sor
        c_next_SOR_sink!(u, omega, sink_mask)
    elseif smoother == :rb_sor
        c_next_SOR_sink_red_black!(u, omega, sink_mask)
    else
        throw(ArgumentError("Unknown smoother '$smoother'. Use :sor or :rb_sor."))
    end
    return nothing
end


function _copy_f64_to_f32!(dst::Matrix{Float32}, src::Matrix{Float64})
    @inbounds for j in axes(src, 2), i in axes(src, 1)
        dst[i, j] = Float32(src[i, j])
    end
    return nothing
end


function _copy_f32_to_f64!(dst::Matrix{Float64}, src::Matrix{Float32})
    @inbounds for j in axes(src, 2), i in axes(src, 1)
        dst[i, j] = Float64(src[i, j])
    end
    return nothing
end


function _mg_smooth_many_gpu!(
    u::Matrix{Float64},
    omega::Float64,
    sink_mask::Matrix{Bool},
    ::Val{:metal},
    c_dev::MtlMatrix{Float32,PrivateStorage},
    sink_dev::MtlMatrix{Bool,PrivateStorage},
    u_host32::Matrix{Float32},
    nsweeps::Int,
)
    _copy_f64_to_f32!(u_host32, u)
    copyto!(c_dev, u_host32)
    @inbounds for _ in 1:nsweeps
        c_next_SOR_sink_metal!(c_dev, Float32(omega), sink_dev)
    end
    copyto!(u_host32, c_dev)
    _copy_f32_to_f64!(u, u_host32)
    u[sink_mask] .= 0.0
    return nothing
end


function _mg_smooth_many_gpu!(
    u::Matrix{Float64},
    omega::Float64,
    sink_mask::Matrix{Bool},
    ::Val{:cuda},
    c_dev::CuArray{Float32,2},
    sink_dev::CuArray{Bool,2},
    u_host32::Matrix{Float32},
    nsweeps::Int,
)
    _copy_f64_to_f32!(u_host32, u)
    copyto!(c_dev, u_host32)
    @inbounds for _ in 1:nsweeps
        c_next_SOR_sink_cuda!(c_dev, Float32(omega), sink_dev)
    end
    copyto!(u_host32, c_dev)
    _copy_f32_to_f64!(u, u_host32)
    u[sink_mask] .= 0.0
    return nothing
end


function _mg_smooth_many_gpu!(
    u::Matrix{Float64},
    omega::Float64,
    sink_mask::Matrix{Bool},
    backend::Symbol,
    c_dev,
    sink_dev,
    u_host32::Matrix{Float32},
    nsweeps::Int,
)
    if nsweeps <= 0
        return nothing
    end
    if backend == :metal || backend == :cuda
        return _mg_smooth_many_gpu!(u, omega, sink_mask, Val(backend), c_dev, sink_dev, u_host32, nsweeps)
    end
    throw(ArgumentError("Unknown backend '$backend' for GPU smoother."))
end


"""
    _max_laplace_residual(u, sink_mask)

Compute the infinity norm of the discrete Laplace residual on interior non-sink
cells, using periodic wrapping in x and interior points in y.
"""
function _max_laplace_residual(u::Matrix{Float64}, sink_mask::Matrix{Bool})::Float64
    ni, nj = size(u)
    max_res = 0.0
    @inbounds for i in 1:ni
        ip = (i == ni) ? 1 : i + 1
        im = (i == 1) ? ni : i - 1
        for j in 2:nj-1
            if sink_mask[i, j]
                continue
            end
            r = abs((u[ip, j] + u[im, j] + u[i, j+1] + u[i, j-1]) - 4.0 * u[i, j])
            if r > max_res
                max_res = r
            end
        end
    end
    return max_res
end


"""
    _laplace_mg_vcycle!(u_levels, mask_levels, level, max_level, omega, smoother, pre_sweeps, post_sweeps, coarse_sweeps, backend, c_dev, sink_dev, u_host32)

Run one recursive multigrid V-cycle for Laplace with sinks using preallocated
hierarchy buffers. Finest-level smoothing can be offloaded to GPU backends.
"""
function _laplace_mg_vcycle!(
    u_levels::Vector{Matrix{Float64}},
    mask_levels::Vector{Matrix{Bool}},
    level::Int,
    max_level::Int,
    omega::Float64,
    smoother::Symbol,
    pre_sweeps::Int,
    post_sweeps::Int,
    coarse_sweeps::Int,
    backend::Symbol,
    c_dev,
    sink_dev,
    u_host32::Union{Nothing,Matrix{Float32}},
)
    u = u_levels[level]
    sink_mask = mask_levels[level]

    if level >= max_level || min(size(u)...) <= 5
        for _ in 1:coarse_sweeps
            _mg_smooth_step_cpu!(u, omega, sink_mask, smoother)
        end
        _apply_bc_and_sink!(u, sink_mask)
        return nothing
    end

    if level == 1 && backend != :cpu
        _mg_smooth_many_gpu!(u, omega, sink_mask, backend, c_dev, sink_dev, u_host32::Matrix{Float32}, pre_sweeps)
    else
        for _ in 1:pre_sweeps
            _mg_smooth_step_cpu!(u, omega, sink_mask, smoother)
        end
    end

    u_coarse = u_levels[level+1]
    sink_mask_coarse = mask_levels[level+1]
    _restrict_avg!(u_coarse, u)
    _coarsen_mask!(sink_mask_coarse, sink_mask)

    _laplace_mg_vcycle!(
        u_levels,
        mask_levels,
        level + 1,
        max_level,
        omega,
        smoother,
        pre_sweeps,
        post_sweeps,
        coarse_sweeps,
        :cpu,
        nothing,
        nothing,
        nothing,
    )

    _prolong_bilinear!(u, u_coarse)
    _apply_bc_and_sink!(u, sink_mask)

    if level == 1 && backend != :cpu
        _mg_smooth_many_gpu!(u, omega, sink_mask, backend, c_dev, sink_dev, u_host32::Matrix{Float32}, post_sweeps)
    else
        for _ in 1:post_sweeps
            _mg_smooth_step_cpu!(u, omega, sink_mask, smoother)
        end
    end
    _apply_bc_and_sink!(u, sink_mask)

    return nothing
end


"""
    laplace_multigrid!(c; sink_indices=nothing, omega=1.6, smoother=:sor, ncycles=8, levels=0, pre_sweeps=2, post_sweeps=2, coarse_sweeps=30, tol=1e-6, backend=:cpu)

Run a multigrid V-cycle solver for Laplace's equation with sink constraints.
Returns the maximum Laplace residual after the final cycle.
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
    backend::Symbol=:cpu,
)::Float64
    if ncycles < 1
        throw(ArgumentError("ncycles must be >= 1"))
    end
    if backend ∉ (:cpu, :metal, :cuda)
        throw(ArgumentError("backend must be one of :cpu, :metal, :cuda"))
    end

    if backend != :cpu && smoother != :rb_sor
        @info "Overriding multigrid smoother to :rb_sor for backend=$backend (GPU smoothing supports RB only)."
        smoother = :rb_sor
    end
    if backend == :cuda && !CUDA.functional()
        error("backend=:cuda requested, but CUDA is not functional in this environment.")
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
    _apply_bc_and_sink!(c, sink_mask)

    # Preallocate hierarchy buffers once per multigrid solve.
    u_levels = Vector{Matrix{Float64}}(undef, nlevels)
    mask_levels = Vector{Matrix{Bool}}(undef, nlevels)
    u_levels[1] = c
    mask_levels[1] = sink_mask
    for level in 2:nlevels
        nfi, nfj = size(u_levels[level-1])
        ci = max(2, cld(nfi, 2))
        cj = max(2, cld(nfj, 2))
        u_levels[level] = zeros(Float64, ci, cj)
        mask_levels[level] = zeros(Bool, ci, cj)
    end

    c_dev = nothing
    sink_dev = nothing
    u_host32 = nothing
    if backend == :metal
        c_dev = MtlMatrix(Matrix{Float32}(c))
        sink_dev = MtlMatrix(sink_mask)
        u_host32 = Matrix{Float32}(undef, size(c)...)
    elseif backend == :cuda
        c_dev = CuArray(Matrix{Float32}(c))
        sink_dev = CuArray(sink_mask)
        u_host32 = Matrix{Float32}(undef, size(c)...)
    end

    max_res = Inf
    for _ in 1:ncycles
        _laplace_mg_vcycle!(
            u_levels,
            mask_levels,
            1,
            nlevels,
            omega,
            smoother,
            pre_sweeps,
            post_sweeps,
            coarse_sweeps,
            backend,
            c_dev,
            sink_dev,
            u_host32,
        )
        max_res = _max_laplace_residual(c, sink_mask)
        if max_res < tol
            break
        end
    end

    return max_res
end

end # module
