module Assignment_2_2

# Import local module
include("../helpers/__init__.jl")

# Diffusion helpers
using .Helpers.DLACore: choose_candidate_monte_carlo
using .Helpers.DLAUtil: run_diffusion_limited_aggregation

DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_2"


function main(; use_GPU::Bool=false, do_bench::Bool=false, do_gif::Bool=false, do_cache::Bool=false, plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR)
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

    p_s::Float64 = 1.0  # Probability of sticking

    @time "Finished diffusion limited aggregation" run_diffusion_limited_aggregation(
        N,
        L,
        tol,
        frames; i_max_conv=i_max,
        omega_sor=omega_sor,
        p_s=p_s,
        candidate_picker=choose_candidate_monte_carlo,
        use_GPU=use_GPU,
        do_gif=do_gif,
        plot_output_dir=plot_output_dir
    )


    return
end

end # module