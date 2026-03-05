module DLAUtil

using Metal: MtlMatrix, PrivateStorage
using ProgressMeter: @showprogress
using Printf: @sprintf
using Plots: heatmap, Plot, GRBackend

include("diffusion.jl")
using .Diffusion: c_anal_2d

include("DLA_core.jl")
using .DLACore: FloatMatrix, diffusion_limited_aggregation_step!

include("savefig.jl")
using .SaveFig: savefig_auto_folder

include("distributed_gif.jl")
using .DistributedGIF: distributed_gif

include("get_heatmap_kwargs.jl")



function superimpose_c_sink(c::FloatMatrix, c_sink::Matrix{Bool})::FloatMatrix
    c_plot = copy(c)
    c_plot[c_sink] .= 1.0  # Cap concentration inside [0.0, 1.0] for better visualization
    return max.(min.(c_plot, 1.0), 0.0)
end


function plot_DLA_frame(cpu_c::FloatMatrix, cpu_sink::Matrix{Bool}; heatmap_kwargs...)
    c_plot = superimpose_c_sink(cpu_c, cpu_sink)
    return heatmap(c_plot'; heatmap_kwargs...)
end



function run_diffusion_limited_aggregation(N::Int, L::Float64, eta::Float64, tol::Float64, frames::Int; i_max_conv::Int=10_000, omega_sor::Float64, use_GPU::Bool=false, do_gif::Bool=false, plot_output_dir::String="plots")
    # Instantiate starting conditions

    # Source
    c_source = zeros(Bool, N, N)
    c_source[:, end] .= 1
    # Sink
    c_sink = zeros(Bool, N, N)
    # Single seed
    c_sink[N÷2, 1] = true
    # Start with equilibrium solution of initial conditions
    c = c_anal_2d(N)

    c_old::Union{MtlMatrix{Float32,PrivateStorage},Nothing} = nothing
    diffs::Union{MtlMatrix{Float32,PrivateStorage},Nothing} = nothing

    if use_GPU
        c_sink = MtlMatrix(c_sink)
        c = MtlMatrix(Matrix{Float32}(c))
        # Pre allocate GPU matrices
        c_old = similar(c)
        diffs = similar(c)
    end

    # Allocate once
    cpu_c = Array(c)
    cpu_sink = Array(c_sink)

    @info "Created initial conditions"

    # Allocate plots vector
    plots = Vector{Plot{GRBackend}}(undef, frames)
    # Fetch plotting kwargs
    heatmap_kwargs = get_heatmap_kwargs(N, L)

    @showprogress "Solving frames" for i in 1:1:frames
        diffusion_limited_aggregation_step!(c, c_sink, c_source, eta, cpu_c, cpu_sink; tol=tol, i_max_conv=i_max_conv, omega_sor=omega_sor, use_GPU=use_GPU, c_old=c_old, diffs=diffs)

        if do_gif
            plots[i] = plot_DLA_frame(cpu_c, cpu_sink; title=@sprintf("Iteration %03d", i), heatmap_kwargs...)
        end
    end

    # Save final state plot
    @time "Saved final state" begin
        p = plot_DLA_frame(cpu_c, cpu_sink; title="Final Frame", heatmap_kwargs...)
        savefig_auto_folder(p, joinpath(plot_output_dir, "diffusion_limited_aggregation_end_N=$N.png"))
    end

    # Save gif of the process
    if do_gif
        @time "Finished gif generation" begin
            distributed_gif(plots, joinpath(plot_output_dir, "diffusion_limited_aggregation_N=$N.gif"); fps=60, do_palette=true, width=900, hwaccel="videotoolbox")
        end
    end

    return
end

end # module