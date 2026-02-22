module Assignment_1_1

# Preperare distributed
using Distributed

# Wave equation helpers
include("helpers/wave_equations.jl")

# Benchmarking
include("helpers/benchmark.jl")

# Plotting
using Plots
include("helpers/distributed_gif.jl")
include("helpers/plot_wave.jl")


initial_condition_0 = (x) -> sin(x)
initial_condition_1 = (x) -> sin(2pi * x)
initial_condition_2 = (x) -> sin(5pi * x)
initial_condition_3 = (x) -> (x > 1 / 5 && x < 2 / 5) ? sin(10pi * x) : 0.0



function main(; do_bench::Bool=false, do_gif::Bool=false)
    if do_bench
        @info "Running benchmarks..."

        # Setup for benchmarking
        N_test = 10
        psi_tst = rand(N_test)

        bench_funcs([wave_equation, wave_equation_inb, wave_equation_vec, wave_equation_dist, wave_equation_simd, wave_equation_avx, wave_equation_dist_avx], psi_tst, 1.0, 1.0, N_test)
    end


    # Simulation parameters
    t_0 = 0.0
    t_f = 1.0
    dt = 0.001
    c = 1.0
    L = 1.0
    N = 100
    x = range(0, L, length=N)
    tvals = range(t_0, step=dt, length=Int((t_f - t_0) / dt))
    ts = [0.0, 0.1, 0.3, 0.8, 1.0, 2.0]


    # Get solutions with Euler
    @info "Running simulations with Euler method..."
    @time "Ran Euler method simulation" begin
        solution_1_euler = propagate_psi(initial_condition_1, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
        solution_2_euler = propagate_psi(initial_condition_2, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
        euler_3 = propagate_psi(initial_condition_3, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt, return_velocity=true)
        velocity_3_euler = euler_3.vs
        solution_3_euler = euler_3.psis
        # Get plots and create animation
        plots = get_wave_plots(t_f, t_0, dt, L, N, solution_1_euler, solution_2_euler, solution_3_euler)
        # Plot multi-timepoint figures for each initial condition
        tp_indices = [clamp(Int(floor(t / dt)) + 1, 1, length(tvals)) for t in ts]
        plot_wave_multi(solution_1_euler, x, "Euler, IC 1", tp_indices; output="plots/ex_1_wave_multi_euler_ic1.png")
        plot_wave_multi(solution_2_euler, x, "Euler, IC 2", tp_indices; output="plots/ex_1_wave_multi_euler_ic2.png")
        plot_wave_multi(solution_3_euler, x, "Euler, IC 3", tp_indices; output="plots/ex_1_wave_multi_euler_ic3.png")
    end

    if do_gif
        @info "Creating animation for Euler method..."
        @time "Created Euler method animation" begin
            distributed_gif(plots, "plots/ex_1_anim_euler.gif", do_palette=true)
        end
    end

    # Get solutions with Leapfrog
    @info "Running simulations with Leapfrog method..."
    @time "Ran Leapfrog method simulation" begin
        solution_1_leapfrog = propagate_psi_leapfrog(initial_condition_1, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
        solution_2_leapfrog = propagate_psi_leapfrog(initial_condition_2, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
        leapfrog_3 = propagate_psi_leapfrog(initial_condition_3, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt, return_velocity=true)
        velocity_3_leapfrog = leapfrog_3.vs
        solution_3_leapfrog = leapfrog_3.psis
        # Get plots and create animation
        plots_leapfrog = get_wave_plots(t_f, t_0, dt, L, N, solution_1_leapfrog, solution_2_leapfrog, solution_3_leapfrog)
    end

    if do_gif
        @info "Creating animation for Leapfrog method..."
        @time "Created Leapfrog method animation" begin
            distributed_gif(plots_leapfrog, "plots/ex_1_anim_leapfrog.gif", do_palette=true)
        end
    end

    @info "Plotting Euler vs Leapfrog energy comparison..."
    energy_euler_shifted = energy_shifted_from_solution(solution_3_euler; c=c, L=L, dt=dt, vs=velocity_3_euler)
    energy_leapfrog_shifted = energy_shifted_from_solution(solution_3_leapfrog; c=c, L=L, dt=dt, vs=velocity_3_leapfrog)
    plot_euler_leapfrog_energy(tvals, energy_euler_shifted, energy_leapfrog_shifted; output="plots/ex_1_energy_euler_vs_leapfrog.png")
end

end
