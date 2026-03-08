module Assignment_2_2

# Import local module
include("../helpers/__init__.jl")

# Diffusion helpers
using .Helpers.DLACore: choose_candidate_monte_carlo
using .Helpers.DLAUtil: run_diffusion_limited_aggregation, run_ps_dimension_experiment, rerender_ps_dimension_plot_from_csv


DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_2"


function main(;
    use_GPU::Bool=false,
    backend::Symbol=(use_GPU ? :metal : :cpu),
    solver::Symbol=:rb_sor,
    mg_ncycles::Int=8,
    mg_levels::Int=0,
    mg_pre_sweeps::Int=2,
    mg_post_sweeps::Int=2,
    mg_coarse_sweeps::Int=30,
    mg_smoother::Symbol=:rb_sor,
    mc_i_max::Int=1_000_000,
    do_ps_dimension_sweep::Bool=false,
    rerender_ps_dimension_plot::Bool=false,
    p_s_values=0.1:0.1:1.0,
    p_s_repeats::Int=30,
    do_bench::Bool=false,
    do_gif::Bool=false,
    do_cache::Bool=false,
    plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR,
)
    N::Int = 100        # Grid size
    L = 1.0             # Physical size of the domain
    omega_sor = 1.91    # Over-relaxation parameter for SOR
    tol = 10^(-3)       # Convergence tolerance
    eta = 1.5           # Some parameter (not specified)
    i_max = 10_000      # Maximum number of iterations
    frames = 1000       # Number of frames to save

    if do_bench
        # pass
    end
    if do_ps_dimension_sweep
        run_ps_dimension_experiment(
            N=N,
            L=L,
            tol=tol,
            frames=frames,
            i_max_conv=i_max,
            omega_sor=omega_sor,
            p_s_values=p_s_values,
            repeats=p_s_repeats,
            backend=backend,
            solver=solver,
            mc_i_max=max(mc_i_max, 10_000_000),
            mg_ncycles=mg_ncycles,
            mg_levels=mg_levels,
            mg_pre_sweeps=mg_pre_sweeps,
            mg_post_sweeps=mg_post_sweeps,
            mg_coarse_sweeps=mg_coarse_sweeps,
            mg_smoother=mg_smoother,
            output_dir=plot_output_dir,
            save_csv=true,
        )
        return
    end
    if rerender_ps_dimension_plot
        rerender_ps_dimension_plot_from_csv(input_dir=plot_output_dir)
        return
    end

    p_s::Float64 = 1.0  # Probability of sticking

    @time "Finished diffusion limited aggregation" run_diffusion_limited_aggregation(
        N,
        L,
        tol,
        frames; i_max_conv=i_max,
        omega_sor=omega_sor,
        solver=solver,
        backend=backend,
        mg_ncycles=mg_ncycles,
        mg_levels=mg_levels,
        mg_pre_sweeps=mg_pre_sweeps,
        mg_post_sweeps=mg_post_sweeps,
        mg_coarse_sweeps=mg_coarse_sweeps,
        mg_smoother=mg_smoother,
        mc_i_max=mc_i_max,
        p_s=p_s,
        candidate_picker=choose_candidate_monte_carlo,
        do_gif=do_gif,
        plot_output_dir=plot_output_dir
    )

    return
end

end # module
