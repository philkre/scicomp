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
            save_every=10,
            periodic_x=true,
            periodic_y=true,
        )
    end
    # Plot final state
    filename_final_state = joinpath(plot_output_dir, "gray_scott_final_spots.png")
    p_final = plot_gray_scott_state(u_hist[end], v_hist[end], t_hist[end]; L=L)
    savefig_auto_folder(p_final, filename_final_state)
    @info "Final state (spots pattern) saved to: $(filename_final_state)"
    # Optionally create GIF animation of the simulation
    if do_gif
        filename_gif = joinpath(plot_output_dir, "gray_scott_spots.gif")
        plots = [plot_gray_scott_state(u_hist[i], v_hist[i], t_hist[i]; L=L) for i in eachindex(t_hist)]
        distributed_gif(plots, filename_gif; fps=60, do_palette=true, width=1200)
        @info "Animation saved to: $filename_gif"
    end

    # Try different parameter regimes
    @info "Running scenario 2: Stripes pattern (different f, k)"
    f2 = 0.022
    k2 = 0.051
    @time "Ran simulation 2" begin
        u_hist2, v_hist2, t_hist2 = simulate_gray_scott(
            N,
            5000.0,
            dt,
            dx;
            Du=Du,
            Dv=Dv,
            f=f2,
            k=k2,
            u_init=u_init,
            v_center=v_center,
            center_size=center_size,
            noise_level=noise_level,
            save_every=10,
            periodic_x=true,
            periodic_y=true,
        )
    end
    # Plot final state for stripes pattern
    filename_final_state_stripes = joinpath(plot_output_dir, "gray_scott_final_stripes.png")
    p_stripes = plot_gray_scott_state(u_hist2[end], v_hist2[end], t_hist2[end]; title_prefix="Gray-Scott (Stripes)")
    savefig_auto_folder(p_stripes, filename_final_state_stripes)
    @info "Final state (stripes pattern) saved to: $(filename_final_state_stripes)"
    # Optionally create GIF animation of the simulation
    if do_gif
        filename_gif_stripes = joinpath(plot_output_dir, "gray_scott_stripes.gif")
        plots_stripes = [plot_gray_scott_state(u_hist2[i], v_hist2[i], t_hist2[i]; title_prefix="Gray-Scott (Stripes)", L=L) for i in eachindex(t_hist2)]
        distributed_gif(plots_stripes, filename_gif_stripes; fps=60, do_palette=true, width=1200)
        @info "Animation saved to: $filename_gif_stripes"
    end

    @info "Running scenario 3: Waves pattern"
    f3 = 0.014
    k3 = 0.050
    @time "Ran simulation 3" begin
        u_hist3, v_hist3, t_hist3 = simulate_gray_scott(
            N,
            5000.0,
            dt,
            dx;
            Du=Du,
            Dv=Dv,
            f=f3,
            k=k3,
            u_init=u_init,
            v_center=v_center,
            center_size=center_size,
            noise_level=noise_level,
            save_every=50
        )
    end
    # Plot final state for waves pattern
    filename_final_state_waves = joinpath(plot_output_dir, "gray_scott_final_waves.png")
    p_waves = plot_gray_scott_state(u_hist3[end], v_hist3[end], t_hist3[end]; title_prefix="Gray-Scott (Waves)")
    savefig_auto_folder(p_waves, filename_final_state_waves)
    @info "Final state (waves pattern) saved to: $(filename_final_state_waves)"
    # Optionally create GIF animation of the simulation
    if do_gif
        filename_gif_waves = joinpath(plot_output_dir, "gray_scott_waves.gif")
        plots_waves = [plot_gray_scott_state(u_hist3[i], v_hist3[i], t_hist3[i]; title_prefix="Gray-Scott (Waves)", L=L) for i in eachindex(t_hist3)]
        distributed_gif(plots_waves, filename_gif_waves; fps=60, do_palette=true, width=1200)
        @info "Animation saved to: $filename_gif_waves"
    end

    return
end

end # module