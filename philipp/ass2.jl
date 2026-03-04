import Pkg
Pkg.instantiate()

include("model.jl")

function update_mask!(c, mask, nu)

    # get all growth candidates
    mask_candidates = []
    candidate_c_sum = 0

    # iterate through all free sites and check if they are adjacent to occupied sites
    for (i, j) in findall(mask .== 0.0)
        # check neighbors        
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ni, nj = i + di, j + dj
            if 1 <= ni <= size(mask, 1) && 1 <= nj <= size(mask, 2)
                if mask[ni, nj] == 1.0
                    # candidate for growth
                    push!(mask_candidates, (ni, nj))
                    # add to sum for probability calculation
                    push!(candidate_c_sum, c[ni, nj]^nu)
                end
            end
        end
    end

    # update mask if prob hits for all candidates
    for (i, j) in mask_candidates
        p = c[i, j]^nu / candidate_c_sum
        if rand() < p
            mask[i, j] = 0.0
        end
    end
end

function run_dla(
    N::Int,
    steps::Int,
    nu::Float64
)
    # init concentration grid
    c = zeros(N, N)
    c[:, 1] .= 0.0
    c[:, end] .= 1.0

    # mask for DLA process (0 = occupied, 1 = free)
    m = ones(N, N)
    init_pos = [ceil(N / 2), ceil(N / 2)]
    m[init_pos...] = 0.0

    for step in 1:steps
        # mask update with DLA rules
        update_mask!(c, m, nu)

        # diffusion step with new mask
        laplace_sor!(c; sink_indices=findall(m .== 0.0))
    end
end