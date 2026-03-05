using Test

include(joinpath(@__DIR__, "..", "main.jl"))

const TEST_OUT = joinpath(@__DIR__, "..", "output", "test_smoke")
mkpath(TEST_OUT)

@testset "Smoke Tests" begin
    @test isdefined(Main, :Model)
    @test isdefined(Main, :DataIO)
    @test isdefined(Main, :Sim)
    @test isdefined(Main, :Plotting)

    @testset "Steady-State Methods" begin
        N = 20
        c0 = zeros(N, N)
        c0[:, 1] .= 0
        c0[:, end] .= 1
        eps = 1e-3

        c_j, its_j, del_j = Main.Sim.run_steadystate(copy(c0), eps; method="jacobi", max_iters=20_000)
        c_g, its_g, del_g = Main.Sim.run_steadystate(copy(c0), eps; method="gauss-seidel", max_iters=20_000)
        c_s, its_s, del_s = Main.Sim.run_steadystate(copy(c0), eps; method="sor", omega=1.8, max_iters=20_000)

        @test size(c_j) == (N, N)
        @test size(c_g) == (N, N)
        @test size(c_s) == (N, N)
        @test its_j > 0 && its_g > 0 && its_s > 0
        @test all(isfinite, del_j)
        @test all(isfinite, del_g)
        @test all(isfinite, del_s)

        sink_idxs = CartesianIndex.([(10, 10), (11, 10), (10, 11), (11, 11)])
        c_sink, its_sink, del_sink = Main.Sim.run_steadystate(
            copy(c0),
            eps;
            method="jacobi",
            sink_indices=sink_idxs,
            max_iters=20_000,
        )
        @test its_sink > 0
        @test all(isfinite, del_sink)
        @test all(c_sink[sink_idxs] .== 0.0)

        built_sink = Main.Sim.sink_builder(N)
        @test !isempty(built_sink)
        @test all(2 <= idx[1] <= N - 1 && 2 <= idx[2] <= N - 1 for idx in built_sink)
        built_circle = Main.Sim.sink_builder(N; shape=:circle)
        built_triangle = Main.Sim.sink_builder(N; shape=:triangle)
        @test !isempty(built_circle)
        @test !isempty(built_triangle)
        @test all(2 <= idx[1] <= N - 1 && 2 <= idx[2] <= N - 1 for idx in built_circle)
        @test all(2 <= idx[1] <= N - 1 && 2 <= idx[2] <= N - 1 for idx in built_triangle)
        @test_throws ErrorException Main.Sim.sink_builder(N; shape=:hexagon)
    end

    @testset "Wave Runs" begin
        L = 1.0
        N = 40
        c = 1.0
        dx = L / N
        dt = 1e-3
        n_steps = 20
        x = 0:dx:L

        psi0 = [sin(2 * pi * xi) for xi in x]
        psis_e = Main.Sim.run_wave(psi0, c, dx, dt, n_steps; method="euler")
        psis_l = Main.Sim.run_wave(psi0, c, dx, dt, n_steps; method="leapfrog")
        psiss = Main.Sim.run_wave_1b(c, dx, dt, n_steps, L; method="leapfrog")

        @test size(psis_e) == (length(x), n_steps)
        @test size(psis_l) == (length(x), n_steps)
        @test length(psiss) == 3
        @test all(size(p) == (length(x), n_steps) for p in psiss)
        @test all(isfinite, psis_e)
        @test all(isfinite, psis_l)
    end

    @testset "Diffusion Runs" begin
        N = 20
        dx = 1.0 / (N - 1)
        dy = dx
        filepath = joinpath(TEST_OUT, "smoke_output_interval.h5")
        isfile(filepath) && rm(filepath)

        Main.Sim.run_diffusion(1.0, N, dy, dx, 1e-4, 8, 2; filepath=filepath)
        c, _, _, steps, times = Main.DataIO.load_output(filepath)

        @test size(c, 1) == N
        @test size(c, 2) == N
        @test size(c, 3) == 4
        @test length(steps) == 4
        @test steps == [2, 4, 6, 8]
        @test times !== nothing
        @test length(times) == 4
    end

    @testset "Omega Sweep" begin
        omegas = collect(1.7:0.1:1.9)
        Ns = [20, 30]
        its, conv, comp = Main.Sim.optimise_omega(1e-3, omegas, Ns; max_iters=5_000, omega_band=0.2)

        @test size(its) == (length(omegas), length(Ns))
        @test size(conv) == size(its)
        @test size(comp) == size(its)
        @test any(comp)
        @test any(conv .& comp)

        sink_idxs = CartesianIndex.([(10, 10), (11, 10), (10, 11), (11, 11)])
        its_s, conv_s, comp_s = Main.Sim.optimise_omega(
            1e-3,
            omegas,
            [20];
            max_iters=2_000,
            omega_band=0.2,
            sink_indices=sink_idxs,
        )
        @test size(its_s) == (length(omegas), 1)
        @test size(conv_s) == size(its_s)
        @test size(comp_s) == size(its_s)
        @test any(comp_s)

        its_b, conv_b, comp_b = Main.Sim.optimise_omega(
            1e-3,
            omegas,
            Ns;
            max_iters=2_000,
            omega_band=0.2,
            sink_indices=Main.Sim.sink_builder,
        )
        @test size(its_b) == (length(omegas), length(Ns))
        @test size(conv_b) == size(its_b)
        @test size(comp_b) == size(its_b)
        @test any(comp_b)
    end

    @testset "Panel Plot" begin
        omegas = collect(1.7:0.1:1.9)
        Ns = [20, 30]
        its, conv, comp = Main.Sim.optimise_omega(1e-3, omegas, Ns; max_iters=5_000, omega_band=0.2)

        outfile = joinpath(TEST_OUT, "omega_sweep_panels_smoke.png")
        Main.Plotting.plot_omega_sweep_panels(omegas, Ns, its, outfile, 5_000, conv, comp)

        @test isfile(outfile)
        @test filesize(outfile) > 0

        omega_out = joinpath(TEST_OUT, "omega_optimisation_vector_masked_smoke.png")
        Main.Plotting.plot_omega_optimisation(
            omegas,
            its[:, end],
            omega_out,
            5_000,
            conv[:, end],
            comp[:, end],
        )
        @test isfile(omega_out)
        @test filesize(omega_out) > 0
    end

    @testset "Steady Plot Sink Outline" begin
        N = 20
        c = zeros(N, N)
        c[:, 1] .= 0
        c[:, end] .= 1
        sink_idxs = CartesianIndex.([(10, 10), (11, 10), (10, 11), (11, 11)])
        c, _, _ = Main.Sim.run_steadystate(copy(c), 1e-3; method="jacobi", sink_indices=sink_idxs, max_iters=20_000)

        outfile = joinpath(TEST_OUT, "steadystate_sink_outline_smoke.png")
        Main.Plotting.plot_steadystate(c, outfile; sink_indices=sink_idxs)

        @test isfile(outfile)
        @test filesize(outfile) > 0

        silly_outfile = joinpath(TEST_OUT, "steadystate_sink_silly_smoke.png")
        Main.Plotting.plot_steadystate(c, silly_outfile; sink_indices=sink_idxs, silly=true)
        @test isfile(silly_outfile)
        @test filesize(silly_outfile) > 0
    end

    @testset "Diffusion HDF5 IO" begin
        N = 20
        dx = 1.0 / (N - 1)
        dy = dx
        filepath = joinpath(TEST_OUT, "smoke_output.h5")
        isfile(filepath) && rm(filepath)

        Main.Sim.run_diffusion(1.0, N, dy, dx, 1e-4, 5, 1; filepath=filepath)
        c, rdx, rdy, steps, times = Main.DataIO.load_output(filepath)

        @test isfile(filepath)
        @test size(c, 1) == N
        @test size(c, 2) == N
        @test size(c, 3) == 5
        @test length(steps) == 5
        @test times !== nothing
        @test length(times) == 5
        @test rdx ≈ dx
        @test rdy ≈ dy
    end
end
