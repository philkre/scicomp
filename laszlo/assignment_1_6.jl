module Assignment_1_6

# Benchmarking
include("helpers/benchmark.jl")

# Diffusion helpers
include("helpers/diffusion.jl")

# Utilities
using LaTeXStrings
using ProgressMeter
using Statistics

# Plotting
include("helpers/savefig.jl")
include("helpers/overlay.jl")
include("helpers/heatmap_kwargs.jl")
using Plots


function solution_at_k(solver::Function, k::Int64, c::Matrix{Float64}, args...; kwargs...)
    for _ in 1:k
        c = solver(c, args...; kwargs...)
    end
    return c
end


function plot_solutions_at_k(solvers::Vector{Function}, labels::Vector{String}, c_0::Matrix{Float64}, k::Int64, args...; D::Float64, L::Float64, N::Int64, kwargs...)
    plot(analytical_sol(1.0, D, L, N), title=L"Solution at $k_i= %$k$", xlabel="y", ylabel="c(y)", linestyle=:dot, label="Analytical", dpi=300)

    for (solver, label) in zip(solvers, labels)
        c = solution_at_k(solver, k, copy(c_0); kwargs...)
        plot!(c[1, :], label=label)
    end

    _savefig("plots/diffusion_y_axis_k_$k.png")
end


function plot_error_at_convergence(solvers::Vector{Function}, labels::Vector{String}, c_0::Matrix{Float64}, args...; D::Float64, L::Float64, N::Int64,
    tol::Float64=1e-6, i_max::Int64=10000, kwargs...)
    anal = analytical_sol(1.0, D, L, N)

    plot(title="Error at convergence", xlabel="y", ylabel="Error")
    for (solver, label) in zip(solvers, labels)
        c, _deltas = solve_until_tol(solver, copy(c_0), tol, i_max)
        plot!(abs.(c[1, :] .- anal), label=label)
    end

    _savefig("plots/diffusion_error_y_axis.png")
end


function get_deltas_equal_iterations(c_0::Matrix{Float64}; tol::Float64=1e-6, omegas::Vector{Float64}=[1.99, 1.93, 1.85])
    # Check omegas has 3 values
    @assert length(omegas) == 3 "Omegas vector must have 3 values, got $(length(omegas))"

    # Get delta for each of the methods as a function of iterations

    deltas_JACOBI::Vector{Float64} = []
    deltas_GAUSS_SEIDEL::Vector{Float64} = []
    deltas_SOR_1::Vector{Float64} = []
    deltas_SOR_2::Vector{Float64} = []
    deltas_SOR_3::Vector{Float64} = []

    c_old_j = copy(c_0)
    c_new_gs = copy(c_0)
    c_new_sor_1 = copy(c_0)
    c_new_sor_2 = copy(c_0)
    c_new_sor_3 = copy(c_0)

    # Run until all methods have converged, allow each method equally many iterations
    while true
        # Jacobi
        c_new_j = c_next_jacobi(c_old_j)
        delta_j = delta(c_old_j, c_new_j)
        c_old_j = c_new_j
        push!(deltas_JACOBI, delta_j)

        # Gauss-Seidel
        c_old_gs = copy(c_new_gs)
        c_next_gauss_seidel!(c_new_gs)
        delta_gs = delta(c_old_gs, c_new_gs)
        push!(deltas_GAUSS_SEIDEL, delta_gs)

        # SOR
        c_old_sor_1 = copy(c_new_sor_1)
        c_old_sor_2 = copy(c_new_sor_2)
        c_old_sor_3 = copy(c_new_sor_3)
        c_next_SOR!(c_new_sor_1, omegas[1])
        c_next_SOR!(c_new_sor_2, omegas[2])
        c_next_SOR!(c_new_sor_3, omegas[3])
        delta_sor_1 = delta(c_old_sor_1, c_new_sor_1)
        delta_sor_2 = delta(c_old_sor_2, c_new_sor_2)
        delta_sor_3 = delta(c_old_sor_3, c_new_sor_3)
        push!(deltas_SOR_1, delta_sor_1)
        push!(deltas_SOR_2, delta_sor_2)
        push!(deltas_SOR_3, delta_sor_3)

        if delta_j < tol && delta_gs < tol && delta_sor_1 < tol && delta_sor_2 < tol && delta_sor_3 < tol
            break
        end
    end

    return deltas_JACOBI, deltas_GAUSS_SEIDEL, deltas_SOR_1, deltas_SOR_2, deltas_SOR_3
end


function plot_deltas_equal_iterations(c_0::Matrix{Float64}; tol::Float64=1e-6, omegas::Vector{Float64}=[1.99, 1.93, 1.85])
    deltas_JACOBI, deltas_GAUSS_SEIDEL, deltas_SOR_1, deltas_SOR_2, deltas_SOR_3 = get_deltas_equal_iterations(c_0; tol=tol, omegas=omegas)



    plot(title="Deltas for equal iterations", xlabel="Iteration", ylabel="Delta", yscale=:log10, dpi=300)
    plot!(deltas_JACOBI, label="Jacobi")
    plot!(deltas_GAUSS_SEIDEL, label="Gauss-Seidel")
    for (omega, deltas_sor) in zip(omegas, [deltas_SOR_1, deltas_SOR_2, deltas_SOR_3])
        plot!(deltas_sor, label=L"SOR $\omega=%$omega$")
    end

    hline!([tol], linestyle=:dot, label=L"\delta = %$tol")

    _savefig("plots/diffusion_deltas_equal_iterations.png")
end


function plot_optimal_omega(c_0::Matrix{Float64}; tol::Float64=1e-6, omegas_stage_1=1.5:0.05:1.90, omegas_stage_2=1.90:0.0001:1.99)
    omegas = vcat(omegas_stage_1, omegas_stage_2)

    k_converge_stage_1 = [get_iteration_count_SOR(c_0, omega, tol) for omega in omegas_stage_1]
    k_converge_stage_2 = [get_iteration_count_SOR(c_0, omega, tol) for omega in omegas_stage_2]
    k_converge = vcat(k_converge_stage_1, k_converge_stage_2)

    plot_omega = plot(omegas, k_converge, title="SOR Iteration Count vs Omega", xlabel=L"\omega", ylabel="Iteration Count", label=L"k_{converge}", dpi=300)
    # Plot vertical line at optimal omega
    k_min = minimum(k_converge)
    optimal_omega = omegas[findfirst(==(k_min), k_converge)]

    vline!(plot_omega, [optimal_omega], linestyle=:dot, label=L"\omega_{opt} = %$optimal_omega")
    _savefig("plots/sor_optimal_omega.png")
end


function get_masks(N::Int64)::Tuple{Vector{Matrix{Bool}},Vector{String}}
    # mask square
    mask_sq = zeros(Bool, N, N)
    mask_sq[20:30, 20:30] .= true

    # mask triangles
    mask_triangle_0 = zeros(Bool, N, N)
    mask_triangle_1 = zeros(Bool, N, N)
    mask_triangle_2 = zeros(Bool, N, N)
    mask_triangle_3 = zeros(Bool, N, N)
    for i in 1:N
        mask_triangle_0[i, 1:max(i - 1, 1)] .= true
        mask_triangle_1[max(1 + Int(N / 2 - i), 1):min(Int(N / 2) + i - 1, N), max(N - i, 1)] .= true
        mask_triangle_2[max(1 + Int(N / 2 - i), 1):min(Int(N / 2) + i - 1, N), max(Int(N / 2) - i, 1)] .= true
        mask_triangle_3[max(1 + Int(N / 2 - i), fld(2N, 5)):min(Int(N / 2) + i - 1, N - fld(2N, 5)), max(Int(N / 2) + 1 - i, fld(2N, 5))] .= true
    end

    # mask circle
    mask_circle = zeros(Bool, N, N)
    center = (N / 2, N / 2)
    radius = N / 10
    for i in 1:N
        for j in 1:N
            if (i - center[1])^2 + (j - center[2])^2 < radius^2
                mask_circle[i, j] = true
            end
        end
    end

    return [mask_sq, mask_triangle_3, mask_circle, mask_triangle_0, mask_triangle_1, mask_triangle_2], ["Square", "Triangle 3", "Circle", "Triangle 0", "Triangle 1", "Triangle 2",]
end


function plot_masks(masks::Vector{Matrix{Bool}}, labels::Vector{String}; N::Int64=50, L::Float64=1.0)
    plots = []
    heatmap_kwargs = get_heatmap_kwargs(N, L)

    for (mask, label) in zip(masks, labels)
        push!(plots, heatmap(mask'; title=label, heatmap_kwargs...))
    end

    plot(plots..., layout=(2, 3), size=(1200, 800), dpi=300, suptitle="Masks for diffusion simulation")

    _savefig("plots/diffusion_masks.png")
end


function plot_sink_simulations(c_0::Matrix{Float64}, masks::Vector{Matrix{Bool}}, labels::Vector{String}; N::Int64=50, L::Float64=1.0, tol::Float64=1e-6, omega_sor::Float64=1.9, i_max::Int64=10_000, output::String="plots/diffusion_sinks.png")
    x = range(0, stop=L, length=N)
    y = range(0, stop=L, length=N)

    nplots = length(masks)
    ncols = min(3, max(1, nplots))
    nrows = cld(nplots, ncols)
    p_all = plot(
        layout=(nrows, ncols),
        size=(1200, 800),
        dpi=300,
        show=false,
        margin=2Plots.mm,
        top_margin=1Plots.mm,
        bottom_margin=2Plots.mm,
    )

    # Pass 1: render heatmaps only.
    for (i, (mask, label)) in enumerate(zip(masks, labels))
        c, _deltas = solve_until_tol(c_next_SOR_sink!, c_0, tol, i_max, omega_sor, mask)
        heatmap!(
            p_all,
            x,
            y,
            c';
            subplot=i,
            title=label,
            xlabel="x",
            ylabel="y",
            aspect_ratio=1,
            clims=(0, 1),
            xlims=(0, L),
            ylims=(0, L),
            widen=false,
            yflip=false,
            dpi=300,
        )
    end

    # Pass 2: overlay sink images on the final grid plot.
    for (i, mask) in enumerate(masks)
        _overlay_image!(p_all, x, y, mask; subplot_idx=i, image_path="input/sink", scale=0.12)
    end

    output_path = _savefig(p_all, output)
    @info "Saved sink simulation plot" output_path
    return output_path
end


function plot_iteration_count_sinks(c_0::Matrix{Float64}, omegas::Vector{Float64}, masks::Vector{Matrix{Bool}}, labels::Vector{String}; N::Int64=50, tol::Float64=1e-6, i_max::Int64=10_000)
    # Start plot
    plot(title="SOR Iteration Count vs Omega", xlabel=L"\omega", ylabel="Iteration Count", label=L"k_{converge}", dpi=300)

    # Plot iteration count for each omega and mask
    @showprogress for (mask, label) in zip([zeros(Bool, N, N), masks...], ["No sink", labels...])
        k_converge = [length(solve_until_tol(c_next_SOR_sink!, c_0, tol, i_max, omega, mask; quiet=true)[2]) for omega in omegas]

        plot!(omegas, k_converge, label=label)
    end

    _savefig("plots/sor_omega_sinks.png")
end


function plot_simulation_insulators(c_0::Matrix{Float64}, masks::Vector{Matrix{Bool}}, labels::Vector{String}; N::Int64=50, L::Float64=1.0, omega::Float64=1.85, tol::Float64=1e-6, i_max::Int64=10_000, output::String="plots/diffusion_insulators.png")
    x = range(0, stop=L, length=N)
    y = range(0, stop=L, length=N)

    nplots = length(masks)
    ncols = min(3, max(1, nplots))
    nrows = cld(nplots, ncols)
    p_all = plot(
        layout=(nrows, ncols),
        size=(1200, 800),
        dpi=300,
        show=false,
        margin=2Plots.mm,
        top_margin=1Plots.mm,
        bottom_margin=2Plots.mm,
    )

    # Pass 1: render heatmaps only.
    for (i, (mask, label)) in enumerate(zip(masks, labels))
        c, _deltas = solve_until_tol(c_next_SOR_sink_insulate!, c_0, tol, i_max, omega; (insulate_mask = mask))
        heatmap!(
            p_all,
            x,
            y,
            c';
            subplot=i,
            title=label,
            xlabel="x",
            ylabel="y",
            aspect_ratio=1,
            clims=(0, 1),
            xlims=(0, L),
            ylims=(0, L),
            widen=false,
            yflip=false,
            top_margin=1Plots.mm,
            bottom_margin=2Plots.mm,
            dpi=150,
            legend=true,
        )
    end

    # Pass 2: overlay sink image on final grid.
    for (i, mask) in enumerate(masks)
        _overlay_image!(p_all, x, y, mask; subplot_idx=i, image_path="input/isolator", scale=0.1)
    end

    output_path = _savefig(p_all, output)
    @info "Saved insulator simulation plot" output_path
    return output_path
end



function main(; do_bench=false)
    if do_bench
        # Setup for benchmarking
        N = 50
        c_tst = zeros(N, N)
        c_tst[:, end] .= 1.0
        bench_funcs([c_next_jacobi, c_next_gauss_seidel!, c_next_SOR!], c_tst)
    end

    # Parameters
    N = 50
    L = 1.0
    D = 1.0
    c_0 = zeros(N, N)
    c_0[:, end] .= 1.0
    tol = 10^(-6)
    k = 100
    omegas_test = [1.99, 1.93, 1.85]
    omegas_stage_1_sinks = 1.5:0.01:1.95
    omegas_stage_2_sinks = 1.95:0.01:1.95
    omegas_sinks = vcat(omegas_stage_1_sinks, omegas_stage_2_sinks)


    # Plot solution at k iterations for all methods
    @info "Plotting solutions at k=$k iterations..."
    plot_solutions_at_k([c_next_jacobi, c_next_gauss_seidel!, c_next_SOR!], ["Jacobi", "Gauss-Seidel", "SOR"], c_0, k; D=D, L=L, N=N)

    # Plot error at convergence for all methods
    @info "Plotting error at convergence..."
    plot_error_at_convergence([c_next_jacobi, c_next_gauss_seidel!, c_next_SOR!], ["Jacobi", "Gauss-Seidel", "SOR"], c_0; D=D, L=L, N=N, tol=tol)

    # Plot deltas until for equal max iterations
    @info "Plotting deltas for equal max iterations..."
    plot_deltas_equal_iterations(c_0; tol=tol, omegas=omegas_test)

    # Plot optimal omega
    @info "Plotting optimal omega..."
    @time "Optimal omega found" plot_optimal_omega(c_0; tol=tol, omegas_stage_1=1.5:0.05:1.90, omegas_stage_2=1.90:0.0001:1.99)

    # Generate masks for diffusion simulation
    @info "Generating masks..."
    masks, labels = get_masks(N)

    # Plot masks
    @info "Plotting masks..."
    plot_masks(masks, labels; N=N, L=L)

    # Plot sink simulations
    @info "Plotting sink simulations..."
    plot_sink_simulations(c_0, masks, labels; N=N, L=L, tol=tol, omega_sor=omegas_test[3], i_max=10_000)

    # Plot iteration count for different omegas with sinks
    @info "Plotting iteration count for different omegas with sinks..."
    plot_iteration_count_sinks(c_0, omegas_sinks, masks, labels; N=N, tol=tol, i_max=10_000)

    # Plot Simulation with insulators
    @info "Plotting simulation with insulators..."
    plot_simulation_insulators(c_0, masks, labels; N=N, L=L, omega=omegas_test[3], tol=tol, i_max=10_000)
end

end
