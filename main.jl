using Plots
default(fontfamily="Computer Modern")

function wave1d!(Psi, a, c, dx)
    """
    Discretisation of the 1D wave equation: d^2 Psi / dt^2 = c^2 * d^2 Psi / dx^2
    """
    # boundaries
    a[1] = 0
    a[end] = 0

    @inbounds for i in 2:length(Psi)-1
        a[i] = c^2 * (Psi[i+1] - 2 * Psi[i] + Psi[i-1]) / dx^2
    end
end

function euler_step!(Psi, a, v, c, dx, dt)
    """
    Performs one time step using the Euler method. Updates both Psi and its velocity v.
    """

    # update a
    wave1d!(Psi, a, c, dx)

    # update Psi and v
    @inbounds for i in 2:length(Psi)-1
        v[i] += a[i] * dt
        Psi[i] += v[i] * dt
    end

    # boundary conditions
    Psi[1] = 0
    Psi[end] = 0
    v[1] = 0
    v[end] = 0
end

function leapfrog_step!(Psi, a, v, c, dx, dt)
    """
    Performs one time step using the leapfrog method. Updates both Psi and its velocity v.
    """

    # update a
    wave1d!(Psi, a, c, dx)

    @inbounds for i in 2:length(Psi)-1
        # half-step v and psi
        v[i] += 0.5 * dt * a[i]
        Psi[i] += v[i] * dt
    end
    # boundary condition
    Psi[1] = 0
    Psi[end] = 0

    # update a
    wave1d!(Psi, a, c, dx)

    @inbounds for i in 2:length(Psi)-1
        # final half-step v
        v[i] += 0.5 * dt * a[i]
    end

    v[1] = 0
    v[end] = 0

end

function run(Psi, c, dx, dt, n_steps; method="euler")
    """Runs the wave simulation for a given initial condition Psi and parameters. 
    Returns an array of Psi at each time step.
    """

    # init
    N = length(Psi)
    v = zeros(N)
    a = zeros(N)

    # array that holds the solution at each time step (for plotting)
    psis = zeros(N, n_steps)

    @inbounds for n in 1:n_steps
        if method == "euler"
            euler_step!(Psi, a, v, c, dx, dt)
        elseif method == "leapfrog"
            leapfrog_step!(Psi, a, v, c, dx, dt)
        end
        psis[:, n] = Psi
    end
    return psis
end


function plotl(psiss, x, title)
    """Plots the wave function Psi for different initial conditions at the final time step."""
    for (i, psis) in enumerate(psiss)
        plot!(x, psis, label="\\Psi_$i")
    end
    xlabel!("x")
    ylabel!("Psi")
    title!(title)
    savefig("figure_1A.png")
end

function run_1b(c, dx, dt, n_steps, L; method="euler")
    """Runs the wave simulation for questions 1 b)"""

    # grid
    x = 0:dx:L

    # initial conditions
    Psi_i = [sin(2 * pi * xi) for xi in x]
    Psi_ii = [sin(5 * pi * xi) for xi in x]
    Psi_iii = [(xi < 2 / 5 && xi > 1 / 5) ? sin(10 * pi * xi) : 0 for xi in x]

    # sim
    psis_i = run(Psi_i, c, dx, dt, n_steps; method=method)
    psis_ii = run(Psi_ii, c, dx, dt, n_steps; method=method)
    psis_iii = run(Psi_iii, c, dx, dt, n_steps; method=method)

    return [psis_i, psis_ii, psis_iii]
end


function animate_all(psiss, x; fps=30, ylim=(-1, 1), filename="animation_1C.mp4")
    """Creates an animation of the wave function for a given array of psi values. 
    Input format of psiss: psis[i][:, n] gives the value of Psi for the i-th initial condition at time step n.
    Saves as an mp4 file."""
    Nt = size(psiss[1], 2)

    anim = @animate for n in 1:Nt
        # create a new plot for each time step
        p = plot(ylim=ylim, legend=:bottom, size=(1200, 800), show=false)

        # plot each Psi on the same plot
        for (i, psis) in enumerate(psiss)
            plot!(p, x, psis[:, n], label="\\Psi_$i", show=false)
        end
        xlabel!(p, "x")
        ylabel!(p, "\\Psi")
        title!(p, "1D wave equation — timestep $n")
    end

    # generate video
    mp4(anim, filename, fps=fps)
    return nothing
end

function main()
    # params
    L = 1
    N = 100
    c = 1
    dx = L / N
    dt = 0.001
    T = 2.0
    n_steps = Int(floor(T / dt))


    # run 1b
    # psiss = run_1b(c, dx, dt, n_steps, L; method="euler")

    # plot 1b
    # plotl(psiss, 0:dx:L, "1D wave equation: different initial conditions")

    # animate 1c
    # animate_all(psiss, 0:dx:L; fps=1200)

    # leapfrog
    psiss_leapfrog = run_1b(c, dx, dt, n_steps, L; method="leapfrog")
    plotl(psiss_leapfrog, 0:dx:L, "1D wave equation: leapfrog method")
    animate_all(psiss_leapfrog, 0:dx:L; fps=1200, filename="animation_1C_leapfrog.mp4")

end

main()
