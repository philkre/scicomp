module SaveFig

using Plots: savefig

"""
    resolve_output_path(output::String)::String

Resolve a potentially relative output path to an absolute path.
If the path is already absolute, returns it unchanged.
Otherwise, resolves it relative to the current working directory.

# Arguments
- `output::String`: File path (relative or absolute)

# Returns
- `String`: The absolute path
"""
function resolve_output_path(output::String)::String
    if isabspath(output)
        return output
    end
    return normpath(joinpath(pwd(), output))
end


"""
    auto_mkpath(path::String)::Nothing

Ensure the directory for the given file path exists, creating it if necessary.

# Arguments
- `path::String`: File path whose parent directory should be created

# Returns
- `Nothing`
"""
function auto_mkpath(path::String)::Nothing
    dir = dirname(path)
    if !isdir(dir)
        mkpath(dir)
    end
    return nothing
end


"""
    savefig_auto_folder(output::String)

Save the current plot to the specified output path, automatically creating the parent directory if needed.

# Arguments
- `output::String`: Output file path (relative or absolute)

# Returns
- `String`: The absolute path where the plot was saved
"""
function savefig_auto_folder(output::String)
    output_path = resolve_output_path(output)
    # Ensure the output directory exists, creating it if necessary
    auto_mkpath(output_path)
    # Saves current() plot to output path, overwriting if it already exists
    savefig(output_path)
    isfile(output_path) || error("Failed to save plot to $output_path")
    return output_path
end


"""
    savefig_auto_folder(p, output::String)

Save the specified plot to the output path, automatically creating the parent directory if needed.

# Arguments
- `p`: The plot object to save
- `output::String`: Output file path (relative or absolute)

# Returns
- `String`: The absolute path where the plot was saved
"""
function savefig_auto_folder(p, output::String)
    output_path = resolve_output_path(output)
    auto_mkpath(output_path)
    # Saves plot `p` to output path, overwriting if it already exists
    savefig(p, output_path)
    isfile(output_path) || error("Failed to save plot to $output_path")
    return output_path
end

end # module