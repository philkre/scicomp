module Assignment_1_6

# Benchmarking
include("helpers/benchmark.jl")

# Diffusion helpers
include("helpers/diffusion.jl")

# Utilities
using LaTeXStrings

# Plotting
include("helpers/heatmap_kwargs.jl")
using Plots


function solution_at_k(solver::Function, k::Int64, c::Matrix{Float64}, args...; kwargs...)
    for _ in 1:k
        c = solver(c, args...; kwargs...)
    end
    return c
end


function plot_solutions_at_k(solvers::Vector{Function}, labels::Vector{String}, c_0::Matrix{Float64}, k::Int64, args...; D::Float64, L::Float64, N::Int64, kwargs...)
    plot(analytical_sol(1.0, D, L, N), title=L"Solution at $k_i= %$k$", xlabel="y", ylabel="c(y)", linestyle=:dot, label="Analytical", dpi=150)

    for (solver, label) in zip(solvers, labels)
        c = solution_at_k(solver, k, copy(c_0); kwargs...)
        plot!(c[1, :], label=label)
    end

    savefig("plots/diffusion_y_axis_k_$k.png")
end


function plot_error_at_convergence(solvers::Vector{Function}, labels::Vector{String}, c_0::Matrix{Float64}, args...; D::Float64, L::Float64, N::Int64,
    tol::Float64=1e-6, i_max::Int64=10000, kwargs...)
    anal = analytical_sol(1.0, D, L, N)

    plot(title="Error at convergence", xlabel="y", ylabel="Error")
    for (solver, label) in zip(solvers, labels)
        c, _deltas = solve_until_tol(solver, copy(c_0), tol, i_max)
        plot!(abs.(c[1, :] .- anal), label=label)
    end

    savefig("plots/diffusion_error_y_axis.png")
end




function main(; do_bench=false)
    if do_bench
        # Setup for benchmarking
        N = 50
        c_tst = zeros(N, N)
        c_tst[:, end] .= 1.0
        bench_funcs([c_next_jacobi, c_next_gauss_seidel!, c_next_SOR!], c_tst)
    end

    N = 50
    L = 1.0
    D = 1.0
    c_0 = zeros(N, N)
    c_0[:, end] .= 1.0
    tol = 1e-6
    k = 100

    @info "Plotting solutions at k=$k iterations..."
    plot_solutions_at_k([c_next_jacobi, c_next_gauss_seidel!, c_next_SOR!], ["Jacobi", "Gauss-Seidel", "SOR"], c_0, k; D=D, L=L, N=N)

    @info "Plotting error at convergence..."
    plot_error_at_convergence([c_next_jacobi, c_next_gauss_seidel!, c_next_SOR!], ["Jacobi", "Gauss-Seidel", "SOR"], c_0; D=D, L=L, N=N, tol=tol)
end

end