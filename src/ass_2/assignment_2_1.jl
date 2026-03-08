module Assignment_2_1

using Metal: MtlMatrix
using CUDA: CUDA, CuArray
using Plots: heatmap
using BenchmarkTools

# Import local module
include("../helpers/__init__.jl")

# Diffusion helpers
using .Helpers: Diffusion
using .Helpers.Diffusion: solve_until_tol, c_next_SOR_sink!, c_next_SOR_sink_red_black!, solve_until_tol_metal!, c_next_SOR_sink_metal!, solve_until_tol_cuda!, c_next_SOR_sink_cuda!
using .Helpers.DLAUtil: run_diffusion_limited_aggregation, run_dla_scaling_experiment, run_eta_dimension_experiment, rerender_eta_dimension_plot_from_csv

# Plotting
using .Helpers.SaveFig: savefig_auto_folder
using .Helpers: get_heatmap_kwargs


DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_2"


function _require_multigrid_available()
    if !isdefined(Diffusion, :laplace_multigrid!)
        error("Multigrid benchmark requested, but `laplace_multigrid!` is not available in `Helpers.Diffusion`. Merge the multigrid implementation into `src/helpers/diffusion.jl` first.")
    end
end


function _build_mg_kwargs(; mg_cycles::Int, mg_levels::Int, mg_pre::Int, mg_post::Int, mg_coarse::Int, mg_omega::Float64, tol::Float64, smoother::Symbol)
    return (
        omega=mg_omega,
        smoother=smoother,
        ncycles=mg_cycles,
        levels=mg_levels,
        pre_sweeps=mg_pre,
        post_sweeps=mg_post,
        coarse_sweeps=mg_coarse,
        tol=tol,
    )
end


function run_bench(;
    N::Int=100,
    L::Float64=1.0,
    omega::Float64=1.85,
    tol::Float64=10^-6,
    backend::Symbol=:cpu,
    mg_cycles::Int=8,
    mg_levels::Int=0,
    mg_pre::Int=2,
    mg_post::Int=2,
    mg_coarse::Int=30,
    mg_omega::Float64=1.6,
    plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR,
)
    if backend ∉ (:cpu, :metal, :cuda)
        throw(ArgumentError("Invalid backend=$backend. Valid options: :cpu, :metal, :cuda"))
    end
    if backend == :cuda && !CUDA.functional()
        error("backend=:cuda requested, but CUDA is not functional in this environment.")
    end

    # Create initial conditions
    c_0 = zeros(N, N)
    c_0[:, end] .= 1
    sink_mask = zeros(Bool, N, N)
    sink_mask[N÷2, 1] = true
    sink_indices = findall(sink_mask)
    heatmap_kwargs = get_heatmap_kwargs(N, L)
    savefig_auto_folder(heatmap(c_0'; heatmap_kwargs...), joinpath(plot_output_dir, "bench_c_0_heatmap.png"))

    specs = NamedTuple[]

    if backend == :cpu
        push!(specs, (
            label="cpu_sor",
            single_step=() -> begin
                c = copy(c_0)
                c_next_SOR_sink!(c, omega, sink_mask)
            end,
            solve_to_tol=() -> solve_until_tol(c_next_SOR_sink!, copy(c_0), tol, 10_000, omega, sink_mask; quiet=true, track_deltas=false),
            sanity=() -> solve_until_tol(c_next_SOR_sink!, copy(c_0), tol, 10_000, omega, sink_mask; quiet=false, track_deltas=false),
            to_array=(x) -> x,
        ))

        push!(specs, (
            label="cpu_rb_sor",
            single_step=() -> begin
                c = copy(c_0)
                c_next_SOR_sink_red_black!(c, omega, sink_mask)
            end,
            solve_to_tol=() -> solve_until_tol(c_next_SOR_sink_red_black!, copy(c_0), tol, 10_000, omega, sink_mask; quiet=true, track_deltas=false),
            sanity=() -> solve_until_tol(c_next_SOR_sink_red_black!, copy(c_0), tol, 10_000, omega, sink_mask; quiet=false, track_deltas=false),
            to_array=(x) -> x,
        ))

        _require_multigrid_available()
        push!(specs, (
            label="cpu_mg_sor",
            single_step=() -> begin
                c = copy(c_0)
                Diffusion.laplace_multigrid!(
                    c;
                    sink_indices=sink_indices,
                    backend=:cpu,
                    _build_mg_kwargs(; mg_cycles=1, mg_levels=mg_levels, mg_pre=mg_pre, mg_post=mg_post, mg_coarse=mg_coarse, mg_omega=mg_omega, tol=tol, smoother=:sor)...
                )
                c
            end,
            solve_to_tol=() -> begin
                c = copy(c_0)
                Diffusion.laplace_multigrid!(
                    c;
                    sink_indices=sink_indices,
                    backend=:cpu,
                    _build_mg_kwargs(; mg_cycles=mg_cycles, mg_levels=mg_levels, mg_pre=mg_pre, mg_post=mg_post, mg_coarse=mg_coarse, mg_omega=mg_omega, tol=tol, smoother=:sor)...
                )
                c
            end,
            sanity=() -> begin
                c = copy(c_0)
                Diffusion.laplace_multigrid!(
                    c;
                    sink_indices=sink_indices,
                    backend=:cpu,
                    _build_mg_kwargs(; mg_cycles=mg_cycles, mg_levels=mg_levels, mg_pre=mg_pre, mg_post=mg_post, mg_coarse=mg_coarse, mg_omega=mg_omega, tol=tol, smoother=:sor)...
                )
                c
            end,
            to_array=(x) -> x,
        ))
    else
        c_0_gpu = backend == :metal ? MtlMatrix(Matrix{Float32}(c_0)) : CuArray(Matrix{Float32}(c_0))
        sink_mask_gpu = backend == :metal ? MtlMatrix(Bool.(sink_mask)) : CuArray(Bool.(sink_mask))

        if backend == :metal
            push!(specs, (
                label="gpu_rb_sor",
                single_step=() -> begin
                    c = copy(c_0_gpu)
                    c_next_SOR_sink_metal!(c, Float32(omega), sink_mask_gpu)
                end,
                solve_to_tol=() -> solve_until_tol_metal!(c_next_SOR_sink_metal!, copy(c_0_gpu), tol, 10_000, Float32(omega), sink_mask_gpu; quiet=true, track_deltas=false),
                sanity=() -> solve_until_tol_metal!(c_next_SOR_sink_metal!, copy(c_0_gpu), tol, 10_000, Float32(omega), sink_mask_gpu; quiet=false, track_deltas=false),
                to_array=(x) -> Array(x),
            ))
        else
            push!(specs, (
                label="gpu_rb_sor",
                single_step=() -> begin
                    c = copy(c_0_gpu)
                    c_next_SOR_sink_cuda!(c, Float32(omega), sink_mask_gpu)
                end,
                solve_to_tol=() -> solve_until_tol_cuda!(c_next_SOR_sink_cuda!, copy(c_0_gpu), tol, 10_000, Float32(omega), sink_mask_gpu; quiet=true, track_deltas=false),
                sanity=() -> solve_until_tol_cuda!(c_next_SOR_sink_cuda!, copy(c_0_gpu), tol, 10_000, Float32(omega), sink_mask_gpu; quiet=false, track_deltas=false),
                to_array=(x) -> Array(x),
            ))
        end

        _require_multigrid_available()
        @info "GPU multigrid benchmark uses RB smoother (GPU supports RB smoothing only)."
        push!(specs, (
            label="gpu_mg_rb",
            single_step=() -> begin
                c = copy(c_0)
                Diffusion.laplace_multigrid!(
                    c;
                    sink_indices=sink_indices,
                    backend=backend,
                    _build_mg_kwargs(; mg_cycles=1, mg_levels=mg_levels, mg_pre=mg_pre, mg_post=mg_post, mg_coarse=mg_coarse, mg_omega=mg_omega, tol=tol, smoother=:rb_sor)...
                )
                c
            end,
            solve_to_tol=() -> begin
                c = copy(c_0)
                Diffusion.laplace_multigrid!(
                    c;
                    sink_indices=sink_indices,
                    backend=backend,
                    _build_mg_kwargs(; mg_cycles=mg_cycles, mg_levels=mg_levels, mg_pre=mg_pre, mg_post=mg_post, mg_coarse=mg_coarse, mg_omega=mg_omega, tol=tol, smoother=:rb_sor)...
                )
                c
            end,
            sanity=() -> begin
                c = copy(c_0)
                Diffusion.laplace_multigrid!(
                    c;
                    sink_indices=sink_indices,
                    backend=backend,
                    _build_mg_kwargs(; mg_cycles=mg_cycles, mg_levels=mg_levels, mg_pre=mg_pre, mg_post=mg_post, mg_coarse=mg_coarse, mg_omega=mg_omega, tol=tol, smoother=:rb_sor)...
                )
                c
            end,
            to_array=(x) -> x,
        ))
    end

    for spec in specs
        @info "Benchmarking $(spec.label) single-step"
        display(@benchmark $(spec.single_step)())
        print("\n\n")

        @info "Benchmarking $(spec.label) convergence"
        c_new = spec.sanity()
        c_new_cpu = spec.to_array(c_new)
        savefig_auto_folder(
            heatmap(c_new_cpu'; heatmap_kwargs...),
            joinpath(plot_output_dir, "bench_$(spec.label)_equilibrium_heatmap.png"),
        )
        display(@benchmark $(spec.solve_to_tol)())
        print("\n\n")
    end
end


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
    do_bench_dla_scaling::Bool=false,
    do_eta_dimension_sweep::Bool=false,
    rerender_eta_dimension_plot::Bool=false,
    n_values=40:20:200,
    repeats::Int=20,
    frames_bench::Int=200,
    eta_values=0.1:0.1:2.0,
    eta_repeats::Int=30,
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
        run_bench(; N=N, L=L, omega=omega_sor, tol=tol, backend=backend, plot_output_dir=plot_output_dir)
    end
    if do_bench_dla_scaling
        run_dla_scaling_experiment(
            N_values=n_values,
            repeats=repeats,
            L=L,
            tol=tol,
            frames=frames_bench,
            i_max_conv=i_max,
            omega_sor=omega_sor,
            eta=eta,
            backend=backend,
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
    if do_eta_dimension_sweep
        run_eta_dimension_experiment(
            N=N,
            L=L,
            tol=tol,
            frames=frames,
            i_max_conv=i_max,
            omega_sor=omega_sor,
            eta_values=eta_values,
            repeats=eta_repeats,
            backend=backend,
            solver=solver,
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
    if rerender_eta_dimension_plot
        rerender_eta_dimension_plot_from_csv(input_dir=plot_output_dir)
        return
    end

    @time "Finished diffusion limited aggregation" run_diffusion_limited_aggregation(
        N,
        L,
        tol,
        frames
        ;
        i_max_conv=i_max,
        omega_sor=omega_sor,
        solver=solver,
        backend=backend,
        eta=eta,
        mg_ncycles=mg_ncycles,
        mg_levels=mg_levels,
        mg_pre_sweeps=mg_pre_sweeps,
        mg_post_sweeps=mg_post_sweeps,
        mg_coarse_sweeps=mg_coarse_sweeps,
        mg_smoother=mg_smoother,
        do_gif=do_gif,
        plot_output_dir=plot_output_dir
    )

    # TODO: find optimal omega (?not possible due to different size of mask each frame)
    # TODO: experiment with eta

    return
end

end # module
