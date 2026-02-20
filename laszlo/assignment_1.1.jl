module Assignment_1_1

# Preperare distributed
using Distributed

# Optimization and benchmarking
using BenchmarkTools
using Plots

# Wave equation helpers
include("helpers/wave_equations.jl")

# Plotting
include("helpers/distributed_gif.jl")

function bench_wave_equation_funcs(funcs::Vector{Function}=[wave_equation, wave_equation_inb, wave_equation_vec, wave_equation_dist, wave_equation_simd, wave_equation_avx, wave_equation_dist_avx])::Nothing
    # Setup for benchmarking
    N_test = 10
    psi_tst = rand(N_test)

    for func in (funcs)
        println("Benchmarking $func...")
        display(@benchmark $func($psi_tst, 1.0, 1.0, $N_test))
        print("\n\n")
    end
end

initial_condition_0 = (x) -> sin(x)
initial_condition_1 = (x) -> sin(2pi * x)
initial_condition_2 = (x) -> sin(5pi * x)
initial_condition_3 = (x) -> (x > 1 / 5 && x < 2 / 5) ? sin(5pi * x) : 0.0

function get_wave_plots(t_f::Float64, t_0::Float64, dt::Float64, L::Float64, N::Int64, solution_1::Matrix, solution_2::Matrix, solution_3::Matrix)::Vector{Plots.Plot{Plots.GRBackend}}
    x = range(0, L, length=N)
    i_total = Int((t_f - t_0) / dt)
    plots = @distributed (vcat) for i in 1:i_total
        # create a plot with 3 subplots and a custom layout
        plot(x, solution_1[:, i], ylim=(-1, 1), title="Time: $(round(i*dt, digits=2)) s", xlabel="Position along string", ylabel="Displacement", dpi=150)
        plot!(x, solution_2[:, i])
        plot!(x, solution_3[:, i])
    end
    return plots
end

function main(do_bench::Bool=false)
    if do_bench
        bench_wave_equation_funcs()
    end

    # Simulation parameters
    t_0 = 0.0
    t_f = 1.0
    dt = 0.001
    c = 1.0
    L = 1.0
    N = 1000

    # Get solutions with Euler
    solution_1_euler = propagate_psi(initial_condition_1, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
    solution_2_euler = propagate_psi(initial_condition_2, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
    solution_3_euler = propagate_psi(initial_condition_3, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
    # Get plots and create animation
    plots = get_wave_plots(t_f, t_0, dt, L, N, solution_1_euler, solution_2_euler, solution_3_euler)
    distributed_gif(plots, "plots/ex_1_anim_euler.gif"; do_palette=true)

    # Get solutions with Leapfrog
    solution_1_leapfrog = propagate_psi_leapfrog(initial_condition_1, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
    solution_2_leapfrog = propagate_psi_leapfrog(initial_condition_2, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
    solution_3_leapfrog = propagate_psi_leapfrog(initial_condition_3, L=L, N=N, c=c, t_0=t_0, t_f=t_f, dt=dt)
    # Get plots and create animation
    plots_leapfrog = get_wave_plots(t_f, t_0, dt, L, N, solution_1_leapfrog, solution_2_leapfrog, solution_3_leapfrog)
    distributed_gif(plots_leapfrog, "plots/ex_1_anim_leapfrog.gif", do_palette=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    print("Number of workers: ", nprocs(), "\nNumber of CPU threads: ", Sys.CPU_THREADS, "\n")

    print(ARGS)

    main()
end

end