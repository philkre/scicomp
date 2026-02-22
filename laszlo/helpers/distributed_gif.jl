using Distributed
@everywhere using Plots
@everywhere begin
    ENV["GKSwstype"] = "nul"  # Headless mode - no display
end
using FFMPEG


"""
    gif_slow(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifname::String; fps::Int64=30)

Create an animated GIF from a vector of plots using Plots.jl's built-in animation.

# Arguments
- `plots::Vector{Plots.Plot{Plots.GRBackend}}`: Vector of plot objects to animate
- `gifname::String`: Output filename for the GIF
- `fps::Int64`: Frames per second (default: 30)

# Notes
This is a simple but slower method compared to `distributed_gif`. 
Uses Plots.jl's Animation framework.
"""
function gif_slow(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifname::String; fps::Int64=30)
    # Animate the solution and save frames
    mkpath(dirname(gifname))

    anim = Animation()
    for p in plots
        frame(anim, p)
    end
    gif(anim, gifname, fps=fps)
end


"""
    distributed_gif(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifname::String; fps::Int64=30, do_palette=false, width=600)

Create an animated GIF from a vector of plots using distributed computing and FFMPEG.

# Arguments
- `plots::Vector{Plots.Plot{Plots.GRBackend}}`: Vector of plot objects to animate
- `gifname::String`: Output filename for the GIF
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
function distributed_gif(plots::Vector{Plots.Plot{Plots.GRBackend}}, gifname::String; fps::Int64=30, do_palette=false, width=600)
    mkpath(dirname(gifname))

    # Create a temporary directory to store frames
    tmp_dirname = "tmp_gif" * string(rand(1:10000))  # Generate a unique temporary directory name
    mkdir(tmp_dirname)

    i = 0
    @sync @distributed for p in plots
        savefig(p, "$(tmp_dirname)/frame_$(lpad(i, 4, '0')).png")
        i += 1
    end

    # Build full path pattern for ffmpeg
    frame_pattern = joinpath(tmp_dirname, "frame_%04d.png")

    # Use FFMPEG to create GIF
    if do_palette
        palette_path = joinpath(tmp_dirname, "palette.png")
        run(`$(FFMPEG.ffmpeg) -framerate $fps 
            -i $frame_pattern
            -vf "fps=$fps,scale=$width:-1:flags=lanczos,palettegen"
            -y $palette_path`)


        run(`$(FFMPEG.ffmpeg) -framerate $fps 
            -i $frame_pattern
            -i $palette_path
            -y $gifname
            -hwaccel videotoolbox
            -vf "fps=$fps,scale=$width:-1:flags=fast_bilinear" 
            -filter_complex "fps=$fps,scale=$width:-1:flags=lanczos[x];[x][1:v]paletteuse" 
            -gifflags -transdiff
            `)
    else
        run(`$(FFMPEG.ffmpeg) 
            -hwaccel videotoolbox          
            -i $frame_pattern
            -vf "fps=$fps,scale=$width:-1:flags=fast_bilinear" 
            -gifflags -transdiff           
            -y $gifname`)
    end

    # Clean up temporary files
    rm(tmp_dirname; recursive=true)

    # Detect if we are running in a Jupyter notebook and display the GIF
    if Base.invokelatest(isdefined, Main, :IJulia) && Main.IJulia.inited
        display("image/png", read(gifname))
    else
        println("GIF created successfully: $gifname")
    end
end
