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
function distributed_gif(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifpath::String; fps::Int64=30, do_palette=false, width::Int=0, use_ffmpeg::Bool=true, verbose::Int=1)
    # Parse arguments
    verbose_level = (verbose isa Int ? verbose : verbose ? 32 : 16) # "error"
    width = width == 0 ? plots[1].attr[:size][1] : width  # Default to width of first plot if not specified

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
        # Use FFMPEG to create GIF
        if do_palette
            # Generate palette for better quality
            palette_path = joinpath(anim.dir, "palette.bmp")
            run(`$(FFMPEG.ffmpeg) -framerate $fps 
                -i $frame_pattern
                -vf "fps=$fps,scale=$width:-1:flags=lanczos,palettegen"
                -y $palette_path
                -v $verbose_level
                `)


            run(`$(FFMPEG.ffmpeg) -framerate $fps 
                -i $frame_pattern
                -i $palette_path
                -y $gifpath
                -hwaccel videotoolbox
                -vf "fps=$fps,scale=$width:-1:flags=fast_bilinear" 
                -filter_complex "fps=$fps,scale=$width:-1:flags=lanczos[x];[x][1:v]paletteuse" 
                -gifflags -transdiff
                -v $verbose_level
                `)
        else
            run(`$(FFMPEG.ffmpeg) 
                -hwaccel videotoolbox          
                -i $frame_pattern
                -vf "fps=$fps,scale=$width:-1:flags=fast_bilinear" 
                -gifflags -transdiff           
                -y $gifpath
                -v $verbose_level
                `)
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