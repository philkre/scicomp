module Assignment_3_2

using SparseArrays: sparse, SparseMatrixCSC, lu
using Statistics: mean
using ProgressMeter: @showprogress
using Plots
using LoopVectorization

# Import local module
using Helpers.SaveFig: savefig_auto_folder


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


"""
    function create_F_field(x_r, y_r, stepsize; A=10^4, sigma=0.2)
"""
function create_F_field(x_r, y_r, stepsize; A=10^4, sigma=0.2)
    rows = Int(8 / stepsize)
    cols = Int(10 / stepsize)
    F_field = Matrix{Float64}(undef, rows, cols)

    inv_2sigma2 = 1 / (2 * sigma^2)
    @turbo for j in 1:cols, i in 1:rows
        y = (j - 1) * stepsize
        x = (i - 1) * stepsize
        F_field[i, j] = A * exp(-((x - x_r)^2 + (y - y_r)^2) * inv_2sigma2)
    end
    return F_field
end


function generate_walls_mask(width::Float64=10.0, height::Float64=8.0; Nx::Int64=2000, Ny::Int64=1600, wallthickness::Float64=0.15)::Matrix{Int64}
    mask = zeros(Int64, Nx, Ny)
    scale_x = Nx / width
    scale_y = Ny / height

    """Meters to index x-direction (right boundary)"""
    ixr = (x) -> Int(floor(x * scale_x))
    """Meters to index x-direction (left boundary)"""
    ixl = (x) -> Int(ceil(x * scale_x))
    """Meters to index y-direction (top boundary)"""
    iyt = (y) -> Int(floor(y * scale_y))
    """Meters to index y-direction (bottom boundary)"""
    iyb = (y) -> Int(ceil(y * scale_y))

    wt = wallthickness
    wt2 = wallthickness / 2

    # Inner walls
    mask[1:ixl(3)-1, iyb(3 - wt2):iyb(3 + wt2)-1] .= 2 # livingroom horzontal side left of door
    mask[ixl(4):ixl(6)-1, iyb(3 - wt2):iyb(3 + wt2)-1] .= 2 # livingroom horzontal side right of door
    mask[ixl(6 - wt2):ixl(6 + wt2)-1, iyb(3):end] .= 2 # livingroom vertical side 

    mask[ixl(7):end, iyb(3 - wt2):iyb(3 + wt2)-1] .= 2 # bathroom horzontal side
    mask[ixl(7 - wt2):ixl(7 + wt2)-1, 1:iyb(1.5)-1] .= 2 # bathroom vertical side beneath door
    mask[ixl(7 - wt2):ixl(7 + wt2)-1, iyb(2.5):iyb(3)-1] .= 2 # bathroom vertical side above door

    mask[ixl(2.5 - wt2):ixl(2.5 + wt2)-1, 1:iyb(2)] .= 2 # kitchen vertical side

    # Boundaries
    mask[:, 1:iyb(wt)] .= 1 # bottom
    mask[:, end-iyb(wt)+1:end] .= 1 # top
    mask[1:ixl(wt), :] .= 1 # left
    mask[end-ixl(wt)+1:end, :] .= 1 # right

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


function construct_matrix(size_u::Tuple{Int,Int}, mask::Matrix{Int64}; k0::Float64=50.0, h::Float64)::SparseMatrixCSC{ComplexF64,Int64}
    I = Int[]
    J = Int[]
    V = ComplexF64[]
    rows, cols = size_u
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

function build_FDM_wifi_plot(u; i::Union{Int,Nothing}=nothing, do_save::Bool=false, output_dir::String=DEFAULT_PLOT_OUTPUT_DIR)
    U = abs.(u)
    U ./= maximum(U)              # normalize 0..1
    U_dB = 20 .* log10.(U .+ 1e-12)  # avoid log10(0)
    p = heatmap(U_dB; clims=(-60, 0))   # dB scale like the assignment
    set_axes_wifi_plot(p)

    if do_save
        output_path = savefig_auto_folder(p, joinpath(output_dir, "wifi_heatmap_$(i).png"))
        println("Saved heatmap to $output_path")
    else
        display(p)
    end
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


function optimize_locations_along_wall(; do_plot=false, save_plots=false, plot_output_dir::String=DEFAULT_PLOT_OUTPUT_DIR)
    h = 0.01
    u_grid = zeros(Float64, Int(8 / h), Int(10 / h))

    #create mask, will be reused by all runs
    mask = Matrix(generate_walls_mask()')

    # Build and factorize matrix ONCE (A depends only on geometry, not source location)
    @time "A (once)" A = construct_matrix(size(u_grid), mask; k0=16.0, h=h)
    @time "LU factorize (once)" F = lu(A)

    along_wall_locations = [(2.8, 2), (2.8, 3), (2.8, 4), (2.8, 5), (2.8, 6),
        (3, 6.2), (4, 6.2), (5, 6.2), (6.2, 5), (5, 1), (1, 2), (1.5, 9), (7, 9)]

    @showprogress for (idx, loc) in enumerate(along_wall_locations)
        @time "Done optimizing for location $loc" begin
            x_i, y_i = loc
            @time "f_field" f_field = create_F_field(x_i, y_i, h)           # 0.02s
            @time "build_rhs" b = build_rhs(f_field, h)                 # 0.01s
            # back-substitution only, much faster than A \ b
            @time "u_vec" u_vec = F \ b                                 # 1.05s
            @time "reshape" u = reshape(u_vec, size(u_grid))            # 0s
            @time "measurement" measurement = check_signals(u, h)       # 0.00004s
            println("at $loc found: $measurement")

            if do_plot
                @time "build_FDM_wifi_plot" begin
                    build_FDM_wifi_plot(u; i=idx, do_save=save_plots, output_dir=plot_output_dir)      # 3.59s 1st plot, 0.1s subsequent plots
                end
            end
        end
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
        optimize_locations_along_wall(; do_plot=true, save_plots=true, plot_output_dir=plot_output_dir)
    end

    return
end

end # module
