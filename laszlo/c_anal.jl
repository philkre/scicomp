using Distributed
using SpecialFunctions

"Analytical solution for the diffusion equation with c(y=0)=0 and c(y=1)=1."
function c_anal(x::Float64, t::Float64; D::Float64=D, i_max::Int=100)
    denom = 2 * sqrt(D * t)

    function c_anal_i(x::Float64, i::Int)
        return erfc((1 - x + 2i) / denom) - erfc((1 + x + 2i) / denom)
    end

    if i_max > 50_000
        return @distributed (+) for i in 0:i_max
            c_anal_i(x, i)
        end
    else
        return sum(c_anal_i(x, i) for i in 0:i_max)
    end
end
