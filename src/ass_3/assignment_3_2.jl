module Assignment_3_2

using BenchmarkTools
using Plots
using SparseArrays
using BenchmarkTools

# Import local module
include("../helpers/__init__.jl")


DEFAULT_PLOT_OUTPUT_DIR = "plots/ass_3"


"""Wave number"""
k(freq) = (2 * π * freq) / (3 * 10^8)

"""
    function next_u(u_index, h, f_grid, u_grid; k = 50)
    i,j = u_index
    north = (i + 1) > 80 ? u_grid[1,j] : u_grid[i+1,j]
    south = (i - 1) < 1 ? u_grid[80,j] : u_grid[i-1,j]
    east = (j + 1) > 100 ? u_grid[i,1] : u_grid[i,j+1]
    west = (j - 1) < 1 ? u_grid[i,100] : u_grid[i,j-1]
    neighbors = (north + south + east + west)
    return (neighbors - f_grid[i,j] * h^2) / (4 - k^2 * h^2)
end
"""
function create_F_field(x_r, y_r, stepsize; A=10^4, sigma=0.2)
    foo(x, y) = A * exp(-(((x - x_r)^2 + (y - y_r)^2) / (2 * sigma^2)))
    return [foo(x, y) for x in 0:stepsize:8-stepsize, y in 0:stepsize:10-stepsize]
end


function inside_walls_mask()
    discretisations = Int(1 / 0.005)
    mask = zeros(8 * discretisations, 10 * discretisations)

    #inner walls 
    mask[586:615, 1:600] .= 2
    mask[586:615, 800:1199] .= 2
    mask[586:615, 1400:end-1] .= 2

    mask[1:400, 486:515] .= 2
    mask[1:300, 1386:1415] .= 2
    mask[500:599, 1386:1415] .= 2
    mask[601:end, 1186:1215] .= 2

    #edges
    mask[1:30, :] .= 1
    mask[end-29:end, :] .= 1
    mask[:, 1:30] .= 1
    mask[:, end-29:end] .= 1

    return mask
end


function find_mask_val(index, h_grid, mask; h_mask=0.005)
    i, j = index
    scalar = h_grid / h_mask           # e.g. 0.1 / 0.005 = 20.0

    # map coarse cell center to fine index range [1 .. rows_mask]
    # using (i-0.5)*h_grid / h_mask + 0.5 would be more physically accurate,
    # but a simple block mapping is OK:
    i_scaled = clamp(Int(round((i - 0.5) * scalar)), 1, size(mask, 1))
    j_scaled = clamp(Int(round((j - 0.5) * scalar)), 1, size(mask, 2))

    return mask[i_scaled, j_scaled]
end


function construct_matrix(u_grid::Matrix{Float64}, mask; k0::Float64=50.0, h::Float64)
    I = Int[]
    J = Int[]
    V = ComplexF64[]
    rows, cols = size(u_grid)
    N = rows * cols

    n_air = 1.0 + 0.0im
    n_wall = 2.5 + 0.5im    # inner walls (mask == 2)

    # centre of the domain in index space
    ic = (rows + 1) / 2
    jc = (cols + 1) / 2

    for j in 1:cols
        for i in 1:rows
            p = i + (j - 1) * rows
            cell_type = find_mask_val((i, j), h, mask)  # 0: air, 1: outer band, 2: inner wall

            if cell_type == 1
                # -------- Absorbing outer band everywhere (impedance-like) --------
                # pick one neighbour that is more "interior" (closer to centre)
                best_score = Inf
                ii = i
                jj = j
                for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    i2 = i + di
                    j2 = j + dj
                    if 1 <= i2 <= rows && 1 <= j2 <= cols
                        # distance to centre
                        s = (i2 - ic)^2 + (j2 - jc)^2
                        if s < best_score
                            best_score = s
                            ii, jj = i2, j2
                        end
                    end
                end

                q = ii + (jj - 1) * rows

                # Impedance relation: (u_wall - u_int)/h - i*k0*u_wall = 0
                # -> (1/h - i*k0) * u_wall - (1/h) * u_int = 0
                push!(I, p)
                push!(J, p)
                push!(V, (1 / h - im * k0))
                push!(I, p)
                push!(J, q)
                push!(V, -1 / h)


            else
                # -------- Interior (air or inner walls) → standard Helmholtz --------

                n_local = (cell_type == 0) ? n_air : n_wall
                k_local = n_local * k0

                # centre
                push!(I, p)
                push!(J, p)
                push!(V, (k_local^2 * h^2 - 4))

                # neighbours (no periodic wrap)
                if i > 1
                    north_idx = p - 1
                    push!(I, p)
                    push!(J, north_idx)
                    push!(V, 1.0 + 0.0im)
                end
                if i < rows
                    south_idx = p + 1
                    push!(I, p)
                    push!(J, south_idx)
                    push!(V, 1.0 + 0.0im)
                end
                if j > 1
                    west_idx = p - rows
                    push!(I, p)
                    push!(J, west_idx)
                    push!(V, 1.0 + 0.0im)
                end
                if j < cols
                    east_idx = p + rows
                    push!(I, p)
                    push!(J, east_idx)
                    push!(V, 1.0 + 0.0im)
                end
            end
        end
    end

    return sparse(I, J, V, N, N)
end

function build_rhs(f_grid, h::Float64)
    rows, cols = size(f_grid)
    b = zeros(ComplexF64, rows * cols)
    for j in 1:cols, i in 1:rows
        p = i + (j - 1) * rows
        b[p] = f_grid[i, j] * h^2
    end
    return b
end

function set_axes_wifi_plot(fig)
    ymin, ymax = ylims(fig)
    positions = range(ymin, ymax, length=9)
    labels = round.(Int, positions ./ ymax .* 8)
    yticks!(fig, positions, string.(labels))
    xmin, xmax = xlims(fig)
    positions = range(xmin, xmax, length=11)
    labels = round.(Int, positions ./ xmax .* 10)
    xticks!(fig, positions, string.(labels))
    return
end

function build_FDM_wifi_plot(u)
    U = abs.(u)
    U ./= maximum(U)              # normalize 0..1
    U_dB = 20 .* log10.(U .+ 1e-12)  # avoid log10(0)
    p = heatmap(U_dB; clims=(-60, 0))   # dB scale like the assignment
    set_axes_wifi_plot(p)
    display(p)
    return
end

function check_signals(u_matrix, h)
    #measurement locations
    locations = Tuple{Int,Int}[(5, 1), (1, 2), (1, 9), (7, 9)]

    scalar = 1 / h
    radius = 0.05 / h
    if radius < 1
        @warn "chosen h (h is $h) makes it impossible to get a circle with radius 5cm so radius will be larger"
    elseif !isinteger(radius)
        @warn "chosen h (h is $h) is not an int so will be rounded to $(Int(radius)) to create ~5cm r circle"
    end


    total_means = Float64[]
    for loc in locations
        i, j = loc .* scalar
        if !isinteger(i) || !isinteger(j)
            @warn "something is going wrong with scaling the signal measurement locations"
        end
        measurement_mean = circular_mean(u_matrix, (Int(i), Int(j)), radius)
        push!(total_means, measurement_mean)
    end
    return sum(total_means)
end

function circular_mean(A, center::Tuple{Int,Int}, radius::Real)
    rows, cols = size(A)
    cy, cx = center
    vals = Float64[]

    for i in max(1, cy - radius):min(rows, cy + radius),
        j in max(1, cx - radius):min(cols, cx + radius)

        if (i - cy)^2 + (j - cx)^2 <= radius^2

            push!(vals, abs(A[Int(i), Int(j)]))
        end
    end
    return mean(vals)
end


function optimize_locations_along_wall()
    h = 0.01
    u_grid = zeros(Float64, Int(8 / h), Int(10 / h))

    #create mask, will be reused by all runs
    mask = inside_walls_mask()

    along_wall_locations = [(2.8, 2), (2.8, 3), (2.8, 4), (2.8, 5), (2.8, 6),
        (3, 6.2), (4, 6.2), (5, 6.2), (6.2, 5), (5, 1), (1, 2), (1.5, 9), (7, 9)]

    for loc in along_wall_locations
        i, j = loc
        f_field = create_F_field(i, j, h)
        A = construct_matrix(u_grid, mask; k0=16.0, h=h)
        b = build_rhs(f_field, h)
        u_vec = A \ b
        u = reshape(u_vec, size(u_grid))
        measurement = check_signals(u, h)
        println("at $loc found: $measurement")
        build_FDM_wifi_plot(u)
    end
end


function main(;
    do_bench::Bool=false,
    do_cache::Bool=false,
    plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR,
)
    if do_bench
        @time "Done benchmarking" begin
            # TODO
        end
    end

    @time "Done optimizing locations along wall" begin
        optimize_locations_along_wall()
    end

    return
end

end # module
