module Assignment_2_3

# Import local module
include("../helpers/__init__.jl")

# Gray-Scott helpers
using .Helpers.GrayScottCore: simulate_gray_scott

# Plotting
using .Helpers.SaveFig: savefig_auto_folder
using .Helpers.GrayScottUtil: plot_gray_scott_state
using .Helpers.DistributedGIF: distributed_gif


function main(; do_bench::Bool=false, do_gif::Bool=false, do_cache::Bool=false, plot_output_dir::String="plots/ass_2")
    # Simulation parameters (as suggested in the assignment)
    N = 128         # Grid size
    L = 1.0         # Domain size
    dt = 1.0        # Time step
    dx = 1.0        # Spatial step
    Du = 0.16       # Diffusion coefficient for U
    Dv = 0.08       # Diffusion coefficient for V  
    f = 0.035       # Feed rate
    k = 0.060       # Kill rate
    T = 10000.0     # Total simulation time

    # Initial conditions
    u_init = 0.5
    v_center = 0.25
    center_size = 10
    noise_level = 0.01

    # Run simulation
    @info ("Running scenario 1: Standard parameters (spots pattern)")
    @time "Ran simulation 1" begin
        u_hist, v_hist, t_hist = simulate_gray_scott(
            N,
            T,
            dt,
            dx;
            Du=Du,
            Dv=Dv,
            f=f,
            k=k,
            u_init=u_init,
            v_center=v_center,
            center_size=center_size,
            noise_level=noise_level,
            save_every=50,
            periodic_x=true,
            periodic_y=true,
        )
    end

    # Plot final state
    filename_final_state = joinpath(plot_output_dir, "gray_scott_final.png")
    p_final = plot_gray_scott_state(u_hist[end], v_hist[end], t_hist[end]; L=L)
    savefig_auto_folder(p_final, filename_final_state)
    @info "Final state saved to: $(filename_final_state)"

    if do_gif
        filename_gif = joinpath(plot_output_dir, "gray_scott_v.gif")
        plots = [plot_gray_scott_state(u_hist[i], v_hist[i], t_hist[i]; L=L) for i in eachindex(t_hist)]
        distributed_gif(plots, filename_gif; fps=60, do_palette=true, width=1200)
        @info "Animation saved to: $filename_gif"
    end

    return

end

end # module