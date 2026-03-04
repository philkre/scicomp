module DistributedGIF

using Distributed
using Printf: @sprintf
using ProgressMeter
@everywhere using Plots
@everywhere begin
    ENV["GKSwstype"] = "nul"  # Headless mode - no display
end
using FFMPEG

include("savefig.jl")
using .SaveFig: auto_mkpath

"""
    gif_slow(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifpath::String; fps::Int64=30)

Create an animated GIF from a vector of plots using Plots.jl's built-in animation.

# Arguments
- `plots::Vector{Plots.Plot{Plots.GRBackend}}`: Vector of plot objects to animate
- `gifpath::String`: Output filename for the GIF
- `fps::Int64`: Frames per second (default: 30)

# Notes
This is a simple but slower method compared to `distributed_gif`. 
Uses Plots.jl's Animation framework.
"""
function gif_slow(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifpath::String; fps::Int64=30)
    # Animate the solution and save frames
    auto_mkpath(gifpath)  # Ensure output directory exists

    anim = Animation()
    @showprogress "Creating GIF frames..." for p in plots
        frame(anim, p)
    end
    gif(anim, gifpath, fps=fps)
end


"""
    run_ffmpeg(anim_dir::String, fps::Int64, frame_pattern::String, gifpath::String, hwaccel_option::Cmd, do_palette::Bool, verbose_level::Int, width::Int)

Execute FFMPEG command to create a GIF animation from a sequence of frames.

# Arguments
- `anim_dir::String`: Directory containing the animation frames
- `fps::Int64`: Frames per second for the output GIF
- `frame_pattern::String`: Input frame pattern (e.g., "frame_%04d.png")
- `gifpath::String`: Output path for the generated GIF file
- `hwaccel_option::Cmd`: Hardware acceleration command options for FFMPEG
- `do_palette::Bool`: If true, generates a color palette for better quality GIF
- `verbose_level::Int`: FFMPEG verbosity level for logging
- `width::Int`: Width of the output GIF in pixels (height scales automatically)

# Details
When `do_palette` is true, the function performs a two-pass encoding:
1. First pass generates an optimized color palette using `palettegen`
2. Second pass applies the palette using `paletteuse` with Lanczos scaling

When `do_palette` is false, performs single-pass encoding with fast bilinear scaling.

Both methods apply `-gifflags -transdiff` for optimized frame differencing.
"""
function run_ffmpeg(anim_dir::String, fps::Int64, frame_pattern::String, gifpath::String, hwaccel_option::Cmd, do_palette::Bool, verbose_level::Int, width::Int)
    # Use FFMPEG to create GIF
    if do_palette
        # Generate palette for better quality
        palette_path = joinpath(anim_dir, "palette.bmp")
        run(`$(FFMPEG.ffmpeg) -framerate $fps 
            -i $frame_pattern
            -vf "fps=$fps,scale=$width:-1:flags=lanczos,palettegen"
            -y $palette_path
            -v $verbose_level
            $(hwaccel_option)
            `)


        run(`$(FFMPEG.ffmpeg) -framerate $fps 
            -i $frame_pattern
            -i $palette_path
            -y $gifpath
            -vf "fps=$fps,scale=$width:-1:flags=fast_bilinear" 
            -filter_complex "fps=$fps,scale=$width:-1:flags=lanczos[x];[x][1:v]paletteuse" 
            -gifflags -transdiff
            -v $verbose_level
            $(hwaccel_option)
            `)
    else
        run(`$(FFMPEG.ffmpeg) 
            -i $frame_pattern
            -vf "fps=$fps,scale=$width:-1:flags=fast_bilinear" 
            -gifflags -transdiff           
            -y $gifpath
            -v $verbose_level
            $(hwaccel_option)
            `)
    end
end


"""
    distributed_gif(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifpath::String; fps::Int64=30, do_palette=false, width=600)

Create an animated GIF from a vector of plots using distributed computing and FFMPEG.

# Arguments
- `plots::Vector{Plots.Plot{Plots.GRBackend}}`: Vector of plot objects to animate
- `gifpath::String`: Output filename for the GIF
- `fps::Int64`: Frames per second (default: 30)
- `do_palette::Bool`: Whether to generate and use a custom color palette for better quality (default: false)
- `width::Int`: Width of the output GIF in pixels (default: 600)

# Notes
- Uses distributed computing to save frames in parallel
- Leverages FFMPEG for efficient GIF encoding with hardware acceleration
- Automatically creates and cleans up temporary frame directory
- With `do_palette=true`, generates an optimized palette for better color reproduction
- Uses videotoolbox hardware acceleration on macOS when available
"""
function distributed_gif(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifpath::String; fps::Int64=30, do_palette=false, width::Int=nothing, use_ffmpeg::Bool=true, verbose::Bool=false, hwaccel::String="OpenCL")
    # Parse arguments
    verbose_level = (verbose ? 32 : 16) # "error"
    width = isnothing(width) ? plots[1].attr[:size][1] : width  # Default to width of first plot if not specified

    anim = Animation()
    n = length(plots)
    prog = Progress(n, 1, "Saving GIF frames...")

    ch = RemoteChannel(() -> Channel{Int}(n))  # buffer n updates
    # consumer task on master that updates the progress bar
    t = @async begin
        for _ in 1:n
            take!(ch)
            next!(prog)
        end
    end

    # Save frames in parallel using distributed workers
    @sync @distributed for i in 1:n
        p = plots[i]
        # savefig(p, "$(tmp_dirname)/frame_$(lpad(i, pad_length, '0')).png")
        filename = @sprintf "%06d.png" i
        png(p, joinpath(anim.dir, filename))
        put!(ch, 1)
    end
    wait(t)  # Ensure the progress bar task completes before proceeding
    # Make sure to add frames in the correct order 
    for i in 1:n
        filename = @sprintf "%06d.png" i
        push!(anim.frames, joinpath(anim.dir, filename))
    end

    # Build full path pattern for ffmpeg
    frame_pattern = joinpath(anim.dir, "%06d.png")

    if use_ffmpeg
        hwaccel_option::Cmd = hwaccel != "" ? `-hwaccel $hwaccel` : ``
        try
            run_ffmpeg(anim.dir, fps, frame_pattern, gifpath, hwaccel_option, do_palette, verbose_level, width)
        catch e
            try
                @warn "FFMPEG failed with error: $e. Falling back to software encoding without hardware acceleration."
                run_ffmpeg(anim.dir, fps, frame_pattern, gifpath, "", do_palette, verbose_level, width)
            catch e
                @warn "FFMPEG failed with error: $e. Falling back to built-in GIF creation method."
                gif(anim, gifpath, fps=fps)
            end
        end
    else
        # Use native gif-function
        gif(anim, gifpath, fps=fps)
    end

    # Clean up temporary files
    rm(anim.dir; recursive=true)

    # Detect if we are running in a Jupyter notebook and display the GIF
    if Base.invokelatest(isdefined, Main, :IJulia) && Main.IJulia.inited
        return AnimatedGif(gifpath)
    else
        println("GIF created successfully: $gifpath")
    end
end

end # module