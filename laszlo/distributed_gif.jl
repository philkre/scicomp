using Distributed
@everywhere using Plots
@everywhere begin
    ENV["GKSwstype"] = "nul"  # Headless mode - no display
end
using FFMPEG
function distributed_gif(t_f::Float64, t_0::Float64, dt::Float64, L::Float64, N::Int, plots::Vector{Plots.Plot{Plots.GRBackend}}, gifname::String; fps::Int64=30, do_palette=false, width=600)
    # Create a temporary directory to store frames
    tmp_dirname = "tmp_gif" * string(rand(1:10000))  # Generate a unique temporary directory name
    mkdir(tmp_dirname)

    # Animate the solution and save frames
    i_total = Int((t_f - t_0) / dt)
    x = range(0, L, length=N)

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

    display("text/html", "<img src='$(gifname)'>")
end