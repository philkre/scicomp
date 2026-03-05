module Assignment_2_1_sijmen
using Distributed
# Import local module
#include("../../src/helpers/__init__.jl")

function print_test(x = 2)
    println(x)
end



function main(; do_bench::Bool=false, do_gif::Bool=false, do_cache::Bool=false, plot_output_dir::String="plots/ass_2")
    pmap( i -> print( "my summark: $(myid()), val: $i"), 1:100)
    
end

end # module