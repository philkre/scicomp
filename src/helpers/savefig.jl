module SaveFig

using Plots: savefig

function resolve_output_path(output::String)::String
    if isabspath(output)
        return output
    end
    return normpath(joinpath(pwd(), output))
end

function auto_mkpath(path::String)::Nothing
    dir = dirname(path)
    if !isdir(dir)
        mkpath(dir)
    end
    return nothing
end


function savefig_auto_folder(output::String)
    output_path = resolve_output_path(output)
    # Ensure the output directory exists, creating it if necessary
    auto_mkpath(output_path)
    # Saves current() plot to output path, overwriting if it already exists
    savefig(output_path)
    isfile(output_path) || error("Failed to save plot to $output_path")
    return output_path
end


function savefig_auto_folder(p, output::String)
    output_path = resolve_output_path(output)
    auto_mkpath(output_path)
    # Saves plot `p` to output path, overwriting if it already exists
    savefig(p, output_path)
    isfile(output_path) || error("Failed to save plot to $output_path")
    return output_path
end

end # module