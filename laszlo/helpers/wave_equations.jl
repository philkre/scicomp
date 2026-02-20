using Distributed
@everywhere using SharedArrays
@everywhere using LoopVectorization

"""
    wave_equation(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}

Compute the second time derivative of the wave function using the wave equation.

# Arguments
- `psi::Vector{Float64}`: Wave function values at spatial points
- `c::Float64`: Wave speed
- `L::Float64`: Domain length
- `N::Int`: Number of spatial grid points

# Returns
- `Vector{Float64}`: Second time derivative ∂²ψ/∂t² at interior points (length N-2)

# Notes
Basic implementation using array slicing. Applies finite difference approximation
for the spatial second derivative.
"""
function wave_equation(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}
    # Starts from second point to second last point
    d2psi_dt2 = c^2 * (psi[1:end-2] - 2 * psi[2:end-1] + psi[3:end]) / (L / N)^2
    return d2psi_dt2
end


"""
    wave_equation_inb(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}

Compute the second time derivative with @inbounds optimization.

# Arguments
- `psi::Vector{Float64}`: Wave function values at spatial points
- `c::Float64`: Wave speed
- `L::Float64`: Domain length
- `N::Int`: Number of spatial grid points

# Returns
- `Vector{Float64}`: Second time derivative ∂²ψ/∂t² at interior points (length N-2)

# Notes
Uses explicit loop with @inbounds for improved performance by skipping bounds checking.
"""
function wave_equation_inb(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}
    dx2_inv = (N / L)^2
    c2 = c^2
    d2psi_dt2 = similar(psi, length(psi) - 2)
    @inbounds for i in 1:length(d2psi_dt2)
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end


"""
    wave_equation_vec(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}

Compute the second time derivative using vectorized operations.

# Arguments
- `psi::Vector{Float64}`: Wave function values at spatial points
- `c::Float64`: Wave speed
- `L::Float64`: Domain length
- `N::Int`: Number of spatial grid points

# Returns
- `Vector{Float64}`: Second time derivative ∂²ψ/∂t² at interior points (length N-2)

# Notes
Uses @views and broadcast operations for efficient vectorized computation.
"""
function wave_equation_vec(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}
    dx2_inv = (N / L)^2
    c2 = c^2
    @views @. c2 * (psi[1:end-2] - 2 * psi[2:end-1] + psi[3:end]) * dx2_inv
end

"""
    wave_equation_dist(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}

Compute the second time derivative using distributed computing.

# Arguments
- `psi::Vector{Float64}`: Wave function values at spatial points
- `c::Float64`: Wave speed
- `L::Float64`: Domain length
- `N::Int`: Number of spatial grid points

# Returns
- `Vector{Float64}`: Second time derivative ∂²ψ/∂t² at interior points (length N-2)

# Notes
Uses SharedArray and @distributed for parallel computation across workers.
"""
function wave_equation_dist(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}
    n = length(psi) - 2
    d2psi_dt2 = SharedArray{Float64}(n)

    @distributed for i in 1:n
        d2psi_dt2[i] = c^2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) / (L / N)^2
    end

    return Array(d2psi_dt2)
end


"""
    wave_equation_simd(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}

Compute the second time derivative using SIMD optimization.

# Arguments
- `psi::Vector{Float64}`: Wave function values at spatial points
- `c::Float64`: Wave speed
- `L::Float64`: Domain length
- `N::Int`: Number of spatial grid points

# Returns
- `Vector{Float64}`: Second time derivative ∂²ψ/∂t² at interior points (length N-2)

# Notes
Uses @simd for single instruction, multiple data vectorization.
"""
function wave_equation_simd(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}
    n = length(psi) - 2
    d2psi_dt2 = Vector{Float64}(undef, n)
    dx2_inv = (N / L)^2
    c2 = c^2

    @inbounds @simd for i in 1:n
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end


"""
    wave_equation_avx(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}

Compute the second time derivative using AVX vectorization.

# Arguments
- `psi::Vector{Float64}`: Wave function values at spatial points
- `c::Float64`: Wave speed
- `L::Float64`: Domain length
- `N::Int`: Number of spatial grid points

# Returns
- `Vector{Float64}`: Second time derivative ∂²ψ/∂t² at interior points (length N-2)

# Notes
Uses @turbo from LoopVectorization.jl for advanced SIMD optimization with AVX instructions.
"""
function wave_equation_avx(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}
    n = length(psi) - 2
    d2psi_dt2 = Vector{Float64}(undef, n)
    dx2_inv = (N / L)^2
    c2 = c^2

    @inbounds @turbo for i in 1:n
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end


"""
    wave_equation_dist_avx(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}

Compute the second time derivative using distributed computing with AVX optimization.

# Arguments
- `psi::Vector{Float64}`: Wave function values at spatial points
- `c::Float64`: Wave speed
- `L::Float64`: Domain length
- `N::Int`: Number of spatial grid points

# Returns
- `Vector{Float64}`: Second time derivative ∂²ψ/∂t² at interior points (length N-2)

# Notes
Combines distributed computing with @turbo AVX vectorization for maximum performance.
Work is divided among available workers with each using SIMD instructions.
"""
function wave_equation_dist_avx(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)::Vector{Float64}
    n = length(psi) - 2
    d2psi_dt2 = SharedArray{Float64}(n)
    dx2_inv = (N / L)^2
    c2 = c^2

    # Slice the array into chunks for each worker
    @sync for p in workers()
        @async begin
            @fetchfrom p begin
                local start_idx = (p - 1) * div(n, nprocs()) + 1
                local end_idx = min(p * div(n, nprocs()), n) + 1
                @inbounds @turbo for i in start_idx:end_idx
                    d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
                end
            end
        end
    end

    return Array(d2psi_dt2)
end


"""
    propagate_psi(psi_0_f::Function; L::Float64=1.0, N::Int=100, c::Float64=1.0, t_0::Float64=0.0, t_f::Float64=1.0, dt::Float64=0.01)::Matrix{Float64}

Propagate the wave equation in time using Euler's method.

# Arguments
- `psi_0_f::Function`: Initial condition function ψ(x) at t=0
- `L`: Domain length (default: 1)
- `N`: Number of spatial grid points (default: 100)
- `c`: Wave speed (default: 1)
- `t_0`: Initial time (default: 0)
- `t_f`: Final time (default: 1)
- `dt`: Time step size (default: 0.01)

# Returns
- `Matrix{Float64}`: Wave function values ψ(x,t) at all spatial points and time steps (N × n_steps)

# Notes
- Uses Euler's method for time integration
- Assumes the string is initially at rest (∂ψ/∂t = 0 at t=0)
- Fixed boundary conditions: ψ(0,t) = ψ(L,t) = 0
- Uses wave_equation_avx for fast spatial derivative computation
"""
function propagate_psi(psi_0_f::Function; L::Float64=1.0, N::Int=100, c::Float64=1.0, t_0::Float64=0.0, t_f::Float64=1.0, dt::Float64=0.01)::Matrix{Float64}
    # Initial condition at t=0
    psi_x_i = psi_0_f.(range(0, L, length=N))

    # The string is at rest at t=0
    # Array is two elements shorter due to second derivative calculation
    dpsi_dt_i = zeros(N - 2)

    # Boundaries are fixed to zero
    psi_x_i[1] = 0
    psi_x_i[end] = 0

    # Initialize the array to store the results
    n_steps = Int((t_f - t_0) / dt)
    psi_x_t = zeros(N, n_steps)
    psi_x_t[:, 1] = psi_x_i

    # Time propagation using finite difference method
    for n in 1:n_steps-1
        d2psi_dt2 = wave_equation_avx(psi_x_t[:, n], c, L, N)
        # Use Euler's method to update psi and its time derivative
        dpsi_dt_i += d2psi_dt2 * dt
        psi_x_t[2:end-1, n+1] = psi_x_t[2:end-1, n] + dpsi_dt_i * dt
    end

    return psi_x_t
end

"""
    propagate_psi_leapfrog(psi_0_f::Function; L::Float64=1.0, N::Int=100, c::Float64=1.0, t_0::Float64=0.0, t_f::Float64=1.0, dt::Float64=0.01)::Matrix{Float64}

Propagate the wave equation in time using the leapfrog (Verlet) integration method.

# Arguments
- `psi_0_f::Function`: Initial condition function ψ(x) at t=0
- `L`: Domain length (default: 1)
- `N`: Number of spatial grid points (default: 100)
- `c`: Wave speed (default: 1)
- `t_0`: Initial time (default: 0)
- `t_f`: Final time (default: 1)
- `dt`: Time step size (default: 0.01)

# Returns
- `Matrix{Float64}`: Wave function values ψ(x,t) at all spatial points and time steps (N × n_steps)

# Notes
- Uses leapfrog (Verlet) integration for improved accuracy and energy conservation
- Assumes the string is initially at rest (∂ψ/∂t = 0 at t=0)
- Fixed boundary conditions: ψ(0,t) = ψ(L,t) = 0
- Uses wave_equation_avx for fast spatial derivative computation
- More stable than Euler's method for wave equations
"""
function propagate_psi_leapfrog(psi_0_f::Function; L::Float64=1.0, N::Int=100, c::Float64=1.0, t_0::Float64=0.0, t_f::Float64=1.0, dt::Float64=0.01)::Matrix{Float64}
    # Initial condition at t=0
    x_i = psi_0_f.(range(0, L, length=N))

    # The string is at rest at t=0
    # Array is two elements shorter due to second derivative calculation
    v_i_slice = zeros(N - 2)

    # Boundaries are fixed to zero
    x_i[1] = 0
    x_i[end] = 0

    # Initialize the array to store the results
    psi_x_t = zeros(N, Int((t_f - t_0) / dt))
    psi_x_t[:, 1] = x_i

    # Time propagation using finite difference method
    for n in 1:size(psi_x_t, 2)-1
        # Calculate acceleration (second time derivative) at current time step
        a_i = wave_equation_avx(x_i, c, L, N)

        # Update position using Verlet integration (leapfrog)
        x_i_next_slice = x_i[2:end-1] + v_i_slice * dt + 0.5 * a_i * dt^2

        # Store the next position in the results array
        psi_x_t[2:end-1, n+1] = x_i_next_slice

        # Calculate acceleration at the next time step for velocity update
        x_i_next = psi_x_t[:, n+1]
        a_i_next = wave_equation_avx(x_i_next, c, L, N)

        # Next time step for velocity
        v_i_slice += 0.5 * (a_i + a_i_next) * dt

        # Shift current position to next for the next iteration
        x_i = x_i_next

    end

    return psi_x_t
end


