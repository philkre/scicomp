module GrayScottCore

using ProgressMeter: @showprogress
using Printf: @sprintf

include("diffusion.jl")
using .Diffusion: laplacian_2d


"""
    gray_scott_step!(u::Matrix{Float64}, v::Matrix{Float64}, Du::Float64, Dv::Float64, 
                     f::Float64, k::Float64, dt::Float64, dx::Float64; 
                     periodic_x::Bool=true, periodic_y::Bool=true)

Perform one time step of the Gray-Scott reaction-diffusion model.

# Arguments
- `u::Matrix{Float64}`: Concentration field of species U (modified in-place)
- `v::Matrix{Float64}`: Concentration field of species V (modified in-place)  
- `Du::Float64`: Diffusion coefficient for U
- `Dv::Float64`: Diffusion coefficient for V
- `f::Float64`: Feed rate for U
- `k::Float64`: Kill rate for V (total decay rate is f+k)
- `dt::Float64`: Time step size
- `dx::Float64`: Spatial grid spacing
- `periodic_x::Bool`: Use periodic boundaries in x (default: true)
- `periodic_y::Bool`: Use periodic boundaries in y (default: true)

# Equations
- ∂u/∂t = Du*∇²u - uv² + f(1-u)
- ∂v/∂t = Dv*∇²v + uv² - (f+k)v

# Notes
Uses explicit Euler time-stepping. Stability requires dt to satisfy the CFL condition.
"""
function gray_scott_step!(
    u::Matrix{Float64},
    v::Matrix{Float64},
    Du::Float64,
    Dv::Float64,
    f::Float64,
    k::Float64,
    dt::Float64,
    dx::Float64
    ;
    periodic_x::Bool=true,
    periodic_y::Bool=true)
    # Compute Laplacians
    lap_u = laplacian_2d(u, dx; periodic_x=periodic_x, periodic_y=periodic_y)
    lap_v = laplacian_2d(v, dx; periodic_x=periodic_x, periodic_y=periodic_y)

    # Compute reaction terms
    uvv = u .* v .* v  # Reaction term: u*v^2

    # Update concentrations using explicit Euler
    u_new = u .+ dt .* (Du .* lap_u .- uvv .+ f .* (1.0 .- u))
    v_new = v .+ dt .* (Dv .* lap_v .+ uvv .- (f .+ k) .* v)

    # Copy back to original arrays
    copyto!(u, u_new)
    copyto!(v, v_new)

    return
end


"""
    simulate_gray_scott(N::Int, T::Float64, dt::Float64, dx::Float64;
                       Du::Float64=0.16, Dv::Float64=0.08,
                       f::Float64=0.035, k::Float64=0.060,
                       u_init::Float64=0.5, v_center::Float64=0.25,
                       center_size::Int=10, noise_level::Float64=0.01,
                       save_every::Int=100, periodic_x::Bool=true, periodic_y::Bool=true)

Run a Gray-Scott simulation.

# Arguments
- `N::Int`: Grid size (NxN)
- `T::Float64`: Total simulation time
- `dt::Float64`: Time step size
- `dx::Float64`: Spatial grid spacing
- `Du::Float64`: Diffusion coefficient for U (default: 0.16)
- `Dv::Float64`: Diffusion coefficient for V (default: 0.08)
- `f::Float64`: Feed rate (default: 0.035)
- `k::Float64`: Kill rate (default: 0.060)
- `u_init::Float64`: Initial uniform concentration of U (default: 0.5)
- `v_center::Float64`: Initial concentration of V in center square (default: 0.25)
- `center_size::Int`: Size of center square for V initialization (default: 10)
- `noise_level::Float64`: Amount of random noise to add (default: 0.01)
- `save_every::Int`: Save state every N steps (default: 100)
- `periodic_x::Bool`: Periodic boundaries in x (default: true)
- `periodic_y::Bool`: Periodic boundaries in y (default: true)

# Returns
- `Tuple{Vector{Matrix{Float64}}, Vector{Matrix{Float64}}, Vector{Float64}}`: 
  Saved U fields, V fields, and corresponding times
"""
function simulate_gray_scott(N::Int,
    T::Float64,
    dt::Float64,
    dx::Float64
    ;
    Du::Float64=0.16,
    Dv::Float64=0.08,
    f::Float64=0.035,
    k::Float64=0.060,
    u_init::Float64=0.5,
    v_center::Float64=0.25,
    center_size::Int=10,
    noise_level::Float64=0.01,
    save_every::Int=100,
    periodic_x::Bool=true,
    periodic_y::Bool=true)::Tuple{Vector{Matrix{Float64}},Vector{Matrix{Float64}},Vector{Float64}}

    # Initialize concentration fields
    u = fill(u_init, N, N)
    v = zeros(N, N)

    # Add V in center square
    center = div(N, 2)
    half_size = div(center_size, 2)
    v[center-half_size:center+half_size, center-half_size:center+half_size] .= v_center

    # Add noise
    if noise_level > 0
        u .+= noise_level .* (rand(N, N) .- 0.5)
        v .+= noise_level .* (rand(N, N) .- 0.5)
    end

    # Clamp to valid range
    clamp!(u, 0.0, 1.0)
    clamp!(v, 0.0, 1.0)

    # Storage for saved states
    u_history = Matrix{Float64}[]
    v_history = Matrix{Float64}[]
    t_history = Float64[]

    # Save initial state
    push!(u_history, copy(u))
    push!(v_history, copy(v))
    push!(t_history, 0.0)

    # Time evolution
    n_steps = round(Int, T / dt)

    @info "Simulating Gray-Scott model"

    @showprogress for step in 1:n_steps
        gray_scott_step!(u, v, Du, Dv, f, k, dt, dx;
            periodic_x=periodic_x, periodic_y=periodic_y)

        # Save state periodically
        if step % save_every == 0
            push!(u_history, copy(u))
            push!(v_history, copy(v))
            push!(t_history, step * dt)
        end
    end

    return u_history, v_history, t_history
end

end # module