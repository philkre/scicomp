module Sim

using ProgressMeter

import ..Model: diffusion2d!, euler_step!, leapfrog_step!, laplace_jacobi!, laplace_gauss_seidel!, laplace_sor!
import ..DataIO: write_out!

export run_wave, run_wave_1b, run_diffusion, run_steadystate, optimise_omega

function run_wave(
    psi0::AbstractVector{<:Real},
    c::Real,
    dx::Real,
    dt::Real,
    n_steps::Integer;
    method::String="euler",
    timing::Bool=false,
)
    """
    Runs the wave simulation for one initial condition.
    """
    psi = collect(Float64, psi0)
    n = length(psi)
    v = zeros(n)
    a = zeros(n)
    psis = zeros(n, n_steps)

    elapsed = @elapsed begin
        @inbounds for step in 1:n_steps
            if method == "euler"
                euler_step!(psi, a, v, c, dx, dt)
            elseif method == "leapfrog"
                leapfrog_step!(psi, a, v, c, dx, dt)
            else
                error("Unknown method '$method'. Use 'euler' or 'leapfrog'.")
            end
            psis[:, step] .= psi
        end
    end

    if timing
        println("run_wave($method) completed in $(round(elapsed; digits=6)) s")
    end

    return psis
end

function run_wave_1b(c::Real, dx::Real, dt::Real, n_steps::Integer, L::Real; method::String="euler")
    """
    Runs the 1D wave simulation for the three assignment initial conditions.
    """
    x = 0:dx:L
    psi_i = [sin(2 * pi * xi) for xi in x]
    psi_ii = [sin(5 * pi * xi) for xi in x]
    psi_iii = [(xi < 2 / 5 && xi > 1 / 5) ? sin(10 * pi * xi) : 0.0 for xi in x]

    psis_i = run_wave(psi_i, c, dx, dt, n_steps; method=method)
    psis_ii = run_wave(psi_ii, c, dx, dt, n_steps; method=method)
    psis_iii = run_wave(psi_iii, c, dx, dt, n_steps; method=method)

    return [psis_i, psis_ii, psis_iii]
end

function run_diffusion(
    D::Float64,
    N::Int,
    dy::Float64,
    dx::Float64,
    dt::Float64,
    steps::Int,
    write_interval::Int;
    timing::Bool=false,
    progress::Bool=false,
    filepath::String="output/data/output.h5",
)
    """
    Runs the diffusion simulation for a given parameter set.
    """
    stab = 4 * dt * D / (dx * dx)
    @assert stab <= 1 "Stability condition violated: 4*D*dt/dx^2 must be <= 1, is $stab"

    c = zeros(N, N)
    c[:, 1] .= 0
    c[:, end] .= 1

    p = progress ? Progress(steps; desc="Diffusion", barlen=30) : nothing

    mkpath(dirname(filepath))
    if isfile(filepath)
        rm(filepath)
    end

    elapsed = @elapsed begin
        for step in 1:steps
            diffusion2d!(c, D, dx, dt)
            if step % write_interval == 0
                write_out!(c, filepath, step, dx, dy; t=step * dt)
            end
            if progress
                next!(p)
            end
        end
    end

    if timing
        println("run completed in $(round(elapsed; digits=6)) s")
    end

    return nothing
end

function run_steadystate(
    c::Matrix{Float64},
    epsilon::Float64;
    method::String="jacobi",
    omega::Union{Nothing,Float64}=nothing,
    max_iters::Int=1_000_000,
)::Tuple{Matrix{Float64},Int64,Vector{Float64}}
    """
    Runs steady-state iterations until convergence to epsilon step difference.
    """
    if !(method in ("jacobi", "gauss-seidel", "sor"))
        error("Unknown method '$method'. Use 'jacobi', 'gauss-seidel', or 'sor'.")
    end

    omega_eff = method == "sor" ? (isnothing(omega) ? 1.8 : omega) : 1.0
    if method == "sor"
        @assert 0.0 < omega_eff < 2.0 "SOR requires 0 < omega < 2, got $omega_eff"
    end

    # Resolve method-specific iteration once (outside convergence loop).
    iter_step! = if method == "jacobi"
        laplace_jacobi!
    elseif method == "gauss-seidel"
        laplace_gauss_seidel!
    else
        c_local -> laplace_sor!(c_local, omega_eff)
    end

    deltas = Float64[]
    its = 0
    while its < max_iters
        c_prev = copy(c)
        iter_step!(c)
        delta = maximum(abs.(c .- c_prev))
        its += 1
        if !isfinite(delta)
            return c, its, deltas
        end
        if delta < epsilon
            return c, its, deltas
        end
        push!(deltas, delta)
    end

    return c, its, deltas
end

function optimise_omega(
    epsilon::Float64,
    omegas::AbstractVector{Float64},
    Ns::AbstractVector{Int};
    max_iters::Int=200_000,
    omega_band::Float64=0.12,
    omega_min::Float64=1.0,
    omega_max::Float64=1.98,
)::Tuple{Matrix{Int64},BitMatrix,BitMatrix}
    """
    Finds optimal omega for SOR over a well-behaved (omega, N) region.
    Region per N is centered around:
    omega*(N) = 2 / (1 + sin(pi/(N-1)))
    and restricted to [omega*(N)-omega_band, omega*(N)+omega_band],
    additionally clipped to [omega_min, omega_max].
    """
    its_vec = Matrix{Int64}(undef, length(omegas), length(Ns))
    converged = falses(length(omegas), length(Ns))
    computed = falses(length(omegas), length(Ns))
    p = Progress(length(omegas) * length(Ns); desc="Optimising omega", barlen=30)

    lower = similar(Float64.(Ns))
    upper = similar(Float64.(Ns))
    for (j, N) in enumerate(Ns)
        omega_star = 2.0 / (1.0 + sin(pi / (N - 1)))
        lower[j] = max(omega_min, omega_star - omega_band)
        upper[j] = min(omega_max, omega_star + omega_band)
    end

    for (j, N) in enumerate(Ns)
        for (i, omega) in enumerate(omegas)
            if omega < lower[j] || omega > upper[j]
                its_vec[i, j] = 0
                next!(p)
                continue
            end

            computed[i, j] = true
            c_local = zeros(N, N)
            c_local[:, 1] .= 0
            c_local[:, end] .= 1
            _, its, _ = run_steadystate(c_local, epsilon; method="sor", omega=omega, max_iters=max_iters)
            its_vec[i, j] = its
            converged[i, j] = its < max_iters
            next!(p)
        end
    end
    return its_vec, converged, computed
end

end
