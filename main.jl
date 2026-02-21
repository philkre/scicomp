import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Printf
using Plots

include("model.jl")
include("data.jl")
include("sim.jl")
include("plot.jl")


using .Sim: run_diffusion, run_wave, run_wave_1b, run_steadystate, optimise_omega, sink_builder
using .Plotting: plot_animation, plot_profiles, plot_2d_concentration, plot_wave_final, animate_wave_all, plot_steadystate, plot_concentration_profiles_steady, plot_convergence_its, plot_omega_optimisation, plot_omega_sweep_panels, plot_wave_multi_t, plot_euler_leapfrog_energy



function main_wave()
    L = 1.0
    N = 100
    c = 1.0
    dx = L / N
    dt = 0.001
    T = 2.0
    n_steps = Int(floor(T / dt))
    x = 0:dx:L

    # Euler
    psiss = run_wave_1b(c, dx, dt, n_steps, L; method="euler")
    #plot_wave_final(psiss, x, "1D wave equation: different initial conditions"; output="figure_1A.png")
    ts = [0.0, 0.1, 0.3, 0.8, 1.0, 2.0]
    nt = size(psiss[1], 2)
    # Timepoint indices are 1-based in Julia and clamped to available output columns.
    tp_indices = [clamp(Int(floor(t / dt)), 1, nt) for t in ts]
    print(tp_indices)
    for (i, psis) in enumerate(psiss)
        plot_wave_multi_t(psis, x, "", tp_indices; output="figure_1B_ic$(i).png")
    end


    # Leapfrog
    psiss_leapfrog = run_wave_1b(c, dx, dt, n_steps, L; method="leapfrog")

    # compare condition three for leapfrog and euler
    psi_string = [(xi < 2 / 5 && xi > 1 / 5) ? sin(10 * pi * xi) : 0.0 for xi in x]
    run_euler_energy = run_wave(psi_string, c, dx, dt, n_steps; method="euler", track_energy=true)
    run_leapfrog_energy = run_wave(psi_string, c, dx, dt, n_steps; method="leapfrog", track_energy=true)

    # initial energy
    strain0 = diff(psi_string) ./ dx
    e0 = 0.5 * c^2 * dx * sum(abs2, strain0)

    # diff energy
    energy_euler_shifted = vcat(0.0, run_euler_energy.energies .- e0)
    energy_leapfrog_shifted = vcat(0.0, run_leapfrog_energy.energies .- e0)
    tvals = (0:n_steps) .* dt

    # plot
    plot_euler_leapfrog_energy(tvals, energy_euler_shifted, energy_leapfrog_shifted; output="figure_1C_energy.png")

    #plot_wave_final(psiss_leapfrog, x, "1D wave equation: leapfrog method"; output="output/img/figure_1A.png")
    # animate_wave_all(psiss_leapfrog, x; fps=1200, filename="output/img/animation_1C_leapfrog.mp4")

    return nothing
end

function main_diffusion()
    D = 1.0
    Lx = 1.0
    Ly = 1.0
    N = 100
    dy = Ly / (N - 1)
    dx = Lx / (N - 1)
    dt = 0.00001
    T = 1.0
    steps = ceil(Int, T / dt)
    write_interval = 1
    output_data_path = "output/data/output.h5"

    dx_str = @sprintf("%.3f", dx)
    dy_str = @sprintf("%.3f", dy)
    println("Discretisation: dx=$dx_str, dy=$dy_str, dt=$dt, steps=$steps")

    # Simulation
    # run_diffusion(D, N, dy, dx, dt, steps, write_interval; timing=true, progress=true, filepath=output_data_path)

    # Plotting
    # plot_animation(output_data_path; fps=30, filename="output/img/diffusion_anim.mp4")
    # plot_profiles(output_data_path, "output/img/profiles.png"; times=[0.001, 0.01, 0.1, 1.0], average_x=true, D=D)
    # plot_2d_concentration(output_data_path, "output/img/concentration.png", [0.0, 0.001, 0.01, 0.1, 1.0])
end



function main_steadystate()

    N = 100
    L = 1.0
    dx = L / (N - 1)
    c = zeros(N, N)
    c[:, 1] .= 0
    c[:, end] .= 1
    epsilon = 1e-6

    ALL_METHODS = false
    PROFILES = false
    CONVERGENCE = false
    OMEGAS = false
    SINKS = true

    # run iterations

    if ALL_METHODS
        c_jacobi, its_jacobi, deltas_jacobi = run_steadystate(copy(c), epsilon; method="jacobi")
        c_gauss_seidel, its_gauss_seidel, deltas_gauss_seidel = run_steadystate(copy(c), epsilon; method="gauss-seidel")
        omega_sor = 1.5
        c_sor15, its_sor15, deltas_sor15 = run_steadystate(copy(c), epsilon; method="sor", omega=omega_sor)
        omega_sor = 1.9
        c_sor19, its_sor19, deltas_sor19 = run_steadystate(copy(c), epsilon; method="sor", omega=omega_sor)

        if PROFILES
            # plot profiles
            plot_concentration_profiles_steady([c_jacobi, c_gauss_seidel, c_sor15, c_sor19], ["Jacobi", "Gauss-Seidel", "SOR 1.5", "SOR 1.9"], "output/img/steadystate_profiles.png")
            println("Iterations for Jacobi: $its_jacobi")
            println("Iterations for Gauss-Seidel: $its_gauss_seidel")
            println("Iterations for SOR (omega=1.5): $its_sor15")
            println("Iterations for SOR (omega=1.9): $its_sor19")
        end

        if CONVERGENCE
            # plot convergence
            plot_convergence_its(
                [deltas_jacobi, deltas_gauss_seidel, deltas_sor15, deltas_sor19],
                ["Jacobi", "Gauss-Seidel", "SOR 1.5", "SOR 1.9"],
                "output/img/steadystate_convergence.png")
        end

        if OMEGAS
            # find optimal omega
            omegas = 1.7:0.01:1.95
            Ns = 10:10:100
            max_iters_omega = 30_000
            its_sor, conv_sor, comp_sor = optimise_omega(
                epsilon,
                omegas,
                Ns;
                max_iters=max_iters_omega,
                omega_band=0.12,
                omega_min=1.0,
                omega_max=1.98,
            )

            # find function fit for optimal omega vs iterations and plot

            plot_omega_optimisation(omegas,
                its_sor[:, end],
                "output/img/omega_optimisation.png",
                max_iters_omega,
                conv_sor[:, end],
                comp_sor[:, end],
            )

            plot_omega_sweep_panels(
                omegas,
                Ns,
                its_sor,
                "output/img/omega_sweep_panels.png",
                max_iters_omega,
                conv_sor,
                comp_sor,
            )

        end

    end

    if SINKS
        sink_indices = sink_builder(N, shape=:circle)

        c_sink_jacobi, its_sink_jacobi, deltas_sink_jacobi = run_steadystate(copy(c), epsilon; method="jacobi", sink_indices=sink_indices)
        c_sink_gauss_seidel, its_sink_gauss_seidel, deltas_sink_gauss_seidel = run_steadystate(copy(c), epsilon; method="gauss-seidel", sink_indices=sink_indices)
        omega_sor = 1.5
        c_sink_sor15, its_sink_sor15, deltas_sink_sor15 = run_steadystate(copy(c), epsilon; method="sor", omega=omega_sor, sink_indices=sink_indices)
        omega_sor = 1.9
        c_sink_sor19, its_sink_sor19, deltas_sink_sor19 = run_steadystate(copy(c), epsilon; method="sor", omega=omega_sor, sink_indices=sink_indices)

        println("Iterations for Jacobi with sink: $its_sink_jacobi")
        println("Iterations for Gauss-Seidel with sink: $its_sink_gauss_seidel")
        println("Iterations for SOR (omega=1.5) with sink: $its_sink_sor15")
        println("Iterations for SOR (omega=1.9) with sink: $its_sink_sor19")

        #plot_steadystate(c_sink_jacobi, "output/img/steadystate_jacobi_sink_silly.png"; sink_indices=sink_indices, silly=true)

        # investigate omega optimisation with sink
        omegas = 1.7:0.01:1.95
        Ns = 10:10:100
        max_iters_omega = 30_000
        its_sor_sink, conv_sor_sink, comp_sor_sink = optimise_omega(
            epsilon,
            omegas,
            Ns;
            max_iters=max_iters_omega,
            omega_band=0.12,
            omega_min=1.0,
            omega_max=1.98,
            sink_indices=sink_builder,
        )
        plot_omega_sweep_panels(
            omegas,
            Ns,
            its_sor_sink,
            "output/img/omega_sweep_panels_sink.png",
            max_iters_omega,
            conv_sor_sink,
            comp_sor_sink,
        )

        plot_omega_optimisation(omegas,
            its_sor_sink[:, end],
            "output/img/omega_optimisation_sink.png",
            max_iters_omega,
            conv_sor_sink[:, end],
            comp_sor_sink[:, end],
        )

    end
end

function main()
    #main_steadystate()
    main_wave()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
