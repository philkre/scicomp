module Sim

using ProgressMeter

import ..Model: diffusion2d!, euler_step!, leapfrog_step!, laplace_jacobi!, laplace_gauss_seidel!, laplace_sor!
import ..DataIO: write_out!

export run_wave, run_wave_1b, run_diffusion, run_steadystate, optimise_omega, sink_builder

function _wave_total_energy(
    psi::AbstractVector{<:Real},
    v::AbstractVector{<:Real},
    c::Real,
    dx::Real,
)::Float64
    strain = diff(psi) ./ dx
    kinetic = 0.5 * dx * sum(abs2, v)
    potential = 0.5 * c^2 * dx * sum(abs2, strain)
    return kinetic + potential
end

function sink_builder(
    N::Int;
    fraction::Float64=0.2,
    shape::Symbol=:square,
)::Vector{CartesianIndex{2}}
    """
    Builds a centered sink on an N x N grid, restricted to interior points.
    Supported shapes: :square (default), :circle, :triangle.
    The sink extent is scaled by `fraction` of the interior size.
    """
    @assert N >= 3 "sink_builder requires N >= 3, got $N"
    @assert 0.0 < fraction <= 1.0 "sink_builder requires 0 < fraction <= 1, got $fraction"
    if !(shape in (:square, :circle, :triangle))
        error("Unknown sink shape '$shape'. Use :square, :circle, or :triangle.")
    end

    interior_n = N - 2
    sink_side = clamp(round(Int, fraction * interior_n), 1, interior_n)
    sink_start = clamp(fld(N - sink_side, 2) + 1, 2, N - 1)
    sink_end = sink_start + sink_side - 1

    if shape == :square
        return vec(CartesianIndices((sink_start:sink_end, sink_start:sink_end)))
    end

    cx = 0.5 * (sink_start + sink_end)
    cy = 0.5 * (sink_start + sink_end)
    half = 0.5 * sink_side

    sink_idxs = CartesianIndex{2}[]
    for j in sink_start:sink_end
        for i in sink_start:sink_end
            if shape == :circle
                dx = (i - cx) / half
                dy = (j - cy) / half
                if dx * dx + dy * dy <= 1.0
                    push!(sink_idxs, CartesianIndex(i, j))
                end
            elseif shape == :triangle
                # Isosceles triangle pointing up, inscribed in the sink bounding box.
                if sink_side == 1
                    push!(sink_idxs, CartesianIndex(i, j))
                else
                    t = (j - sink_start) / (sink_side - 1)  # 0 at top, 1 at bottom
                    width = max(1.0, 1.0 + t * (sink_side - 1))
                    left = cx - 0.5 * (width - 1.0)
                    right = cx + 0.5 * (width - 1.0)
                    if left <= i <= right
                        push!(sink_idxs, CartesianIndex(i, j))
                    end
                end
            end
        end
    end
    return sink_idxs
end

function run_wave(
    psi0::AbstractVector{<:Real},
    c::Real,
    dx::Real,
    dt::Real,
    n_steps::Integer;
    method::String="euler",
    track_energy::Bool=false,
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
    energies = track_energy ? zeros(n_steps) : nothing

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
            if track_energy
                energies[step] = _wave_total_energy(psi, v, c, dx)
            end
        end
    end

    if timing
        println("run_wave($method) completed in $(round(elapsed; digits=6)) s")
    end

    if track_energy
        return (psis=psis, energies=energies)
    end
    return psis
end

function run_wave_1b(
    c::Real,
    dx::Real,
    dt::Real,
    n_steps::Integer,
    L::Real;
    method::String="euler",
    track_energy::Bool=false,
)
    """
    Runs the 1D wave simulation for the three assignment initial conditions.
    """
    x = 0:dx:L
    psi_i = [sin(2 * pi * xi) for xi in x]
    psi_ii = [sin(5 * pi * xi) for xi in x]
    psi_iii = [(xi < 2 / 5 && xi > 1 / 5) ? sin(10 * pi * xi) : 0.0 for xi in x]

    run_i = run_wave(psi_i, c, dx, dt, n_steps; method=method, track_energy=track_energy)
    run_ii = run_wave(psi_ii, c, dx, dt, n_steps; method=method, track_energy=track_energy)
    run_iii = run_wave(psi_iii, c, dx, dt, n_steps; method=method, track_energy=track_energy)

    if track_energy
        return (
            psiss=[run_i.psis, run_ii.psis, run_iii.psis],
            energies=[run_i.energies, run_ii.energies, run_iii.energies],
        )
    end
    return [run_i, run_ii, run_iii]
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
    sink_indices::Union{Nothing,AbstractVector{Int},AbstractVector{CartesianIndex{2}}}=nothing,
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

    # Convert sink indices once to linear Int indices for faster repeated writes.
    sink_linear::Union{Nothing,Vector{Int}} = if isnothing(sink_indices)
        nothing
    elseif sink_indices isa AbstractVector{Int}
        collect(Int, sink_indices)
    else
        lin = LinearIndices(c)
        [lin[idx] for idx in sink_indices]
    end

    # Resolve method-specific iteration once (outside convergence loop).
    iter_step! = if method == "jacobi"
        c_local -> laplace_jacobi!(c_local; sink_indices=sink_linear)
    elseif method == "gauss-seidel"
        c_local -> laplace_gauss_seidel!(c_local; sink_indices=sink_linear)
    else
        c_local -> laplace_sor!(c_local, omega_eff; sink_indices=sink_linear)
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
    sink_indices::Union{
        Nothing,
        AbstractVector{Int},
        AbstractVector{CartesianIndex{2}},
        Function,
    }=nothing,
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
        sink_for_N = if isnothing(sink_indices)
            nothing
        elseif sink_indices isa Function
            sink_gen = sink_indices(N)
            if isnothing(sink_gen)
                nothing
            elseif sink_gen isa AbstractVector{Int}
                sink_gen
            else
                collect(sink_gen)
            end
        elseif sink_indices isa AbstractVector{Int}
            sink_indices
        else
            collect(sink_indices)
        end

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
            _, its, _ = run_steadystate(
                c_local,
                epsilon;
                method="sor",
                omega=omega,
                sink_indices=sink_for_N,
                max_iters=max_iters,
            )
            its_vec[i, j] = its
            converged[i, j] = its < max_iters
            next!(p)
        end
    end
    return its_vec, converged, computed
end

end
