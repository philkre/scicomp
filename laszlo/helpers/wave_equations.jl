function wave_equation(psi, c, L, N)
    # Starts from second point to second last point
    d2psi_dt2 = c^2 * (psi[1:end-2] - 2 * psi[2:end-1] + psi[3:end]) / (L / N)^2
    return d2psi_dt2
end

function wave_equation_inb(psi, c, L, N)
    dx2_inv = (N / L)^2
    c2 = c^2
    d2psi_dt2 = similar(psi, length(psi) - 2)
    @inbounds for i in 1:length(d2psi_dt2)
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end

function wave_equation_vec(psi, c, L, N)
    dx2_inv = (N / L)^2
    c2 = c^2
    @views @. c2 * (psi[1:end-2] - 2 * psi[2:end-1] + psi[3:end]) * dx2_inv
end

function wave_equation_dist(psi, c, L, N)
    n = length(psi) - 2
    d2psi_dt2 = SharedArray{Float64}(n)

    @distributed for i in 1:n
        d2psi_dt2[i] = c^2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) / (L / N)^2
    end

    return Array(d2psi_dt2)
end

function wave_equation_simd(psi, c, L, N)
    n = length(psi) - 2
    d2psi_dt2 = Vector{Float64}(undef, n)
    dx2_inv = (N / L)^2
    c2 = c^2

    @inbounds @simd for i in 1:n
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end

function wave_equation_avx(psi, c, L, N)
    n = length(psi) - 2
    d2psi_dt2 = Vector{Float64}(undef, n)
    dx2_inv = (N / L)^2
    c2 = c^2

    @inbounds @turbo for i in 1:n
        d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
    end
    return d2psi_dt2
end

function wave_equation_dist_avx(psi, c, L, N)
    n = length(psi) - 2
    d2psi_dt2 = SharedArray{Float64}(n)
    dx2_inv = (N / L)^2
    c2 = c^2

    # Slice the array into chunks for each worker
    @sync for p in workers()
        @async begin
            @fetchfrom p begin
                local start_idx = (p - 1) * div(n, nprocs()) + 1
                local end_idx = min(p * div(n, nprocs()), n) + 1
                @inbounds @turbo for i in start_idx:end_idx
                    d2psi_dt2[i] = c2 * (psi[i] - 2 * psi[i+1] + psi[i+2]) * dx2_inv
                end
            end
        end
    end

    return Array(d2psi_dt2)
end
