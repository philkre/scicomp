using SharedArrays
using Distributed
@everywhere using LoopVectorization


function wave_equation(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)
    # Starts from second point to second last point
    d2psi_dt2 = c^2 * (psi[1:end-2] - 2 * psi[2:end-1] + psi[3:end]) / (L / N)^2
    return d2psi_dt2
end


function wave_equation_inb(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)
    dx2_inv = (N / L)^2
    c2 = c^2
    d2psi_dt2 = similar(psi, length(psi) - 2)
    @inbounds for i in 1:length(d2psi_dt2)
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end


function wave_equation_vec(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)
    dx2_inv = (N / L)^2
    c2 = c^2
    @views @. c2 * (psi[1:end-2] - 2 * psi[2:end-1] + psi[3:end]) * dx2_inv
end

function wave_equation_dist(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)
    n = length(psi) - 2
    d2psi_dt2 = SharedArray{Float64}(n)

    @distributed for i in 1:n
        d2psi_dt2[i] = c^2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) / (L / N)^2
    end

    return Array(d2psi_dt2)
end


function wave_equation_simd(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)
    n = length(psi) - 2
    d2psi_dt2 = Vector{Float64}(undef, n)
    dx2_inv = (N / L)^2
    c2 = c^2

    @inbounds @simd for i in 1:n
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end


function wave_equation_avx(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)
    n = length(psi) - 2
    d2psi_dt2 = Vector{Float64}(undef, n)
    dx2_inv = (N / L)^2
    c2 = c^2

    @inbounds @turbo for i in 1:n
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end


function wave_equation_dist_avx(psi::Vector{Float64}, c::Float64, L::Float64, N::Int)
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


function wave_equation_dist_avx(psi, c, L, N)
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

function propagate_psi(psi_0_f::Function; L=1, N=100, c=1, t_0=0, t_f=1, dt=0.01)
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

function propagate_psi_leapfrog(psi_0_f::Function; L=1, N=100, c=1, t_0=0, t_f=1, dt=0.01)
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


