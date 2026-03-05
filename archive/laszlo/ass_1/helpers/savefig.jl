using FileIO

function _resolve_output_path(output::String)::String
    if isabspath(output)
        return output
    end
    return normpath(joinpath(pwd(), output))
end


function _savefig(output::String)
    output_path = _resolve_output_path(output)
    mkpath(dirname(output_path))
    if isfile(output_path)
        rm(output_path; force=true)
    end
    savefig(output_path)
    isfile(output_path) || error("Failed to save plot to $output_path")
    return output_path
end


function _savefig(p, output::String)
    output_path = _resolve_output_path(output)
    mkpath(dirname(output_path))
    if isfile(output_path)
        rm(output_path; force=true)
    end
    savefig(p, output_path)
    isfile(output_path) || error("Failed to save plot to $output_path")
    return output_path
end

