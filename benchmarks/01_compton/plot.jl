using BenchmarkTools
using CairoMakie

include("utils.jl")

plotpath = "plots"
mkpath(plotpath)  # ensure output directory exists

bench_files = filter(
    x -> endswith(x, ".json") && startswith(x, "bench"),
    readdir("data")
)
sort!(bench_files)

@info "Found benchmark files: $bench_files"

for bench_file in bench_files
    backend_str = match(r"^bench_(.*)\.json$", bench_file).captures[1]
    @info "Plot backend: $backend_str"

    # Load benchmark data robustly
    bench_data = try
        BenchmarkTools.load(joinpath("data", bench_file))
    catch e
        @warn "Failed to load $bench_file, skipping" exception = (e, catch_backtrace())
        continue
    end

    if isempty(bench_data)
        @warn "No benchmark data found in $bench_file, skipping"
        continue
    end

    data = bench_data[1]  # assume one top-level dict as before
    for bench in sort!(collect(keys(data)))
        @info "Benchmark: $bench"
        bench_result = data[bench]

        dtypes = sort!(collect(keys(bench_result)))
        ncols = min(length(dtypes), 3)  # up to 3 per row
        nrows = cld(length(dtypes), ncols)

        # Size scales with number of rows/columns
        f = Figure(size = (400 * ncols + 200, 400 * nrows + 100))

        axes = Axis[]  # store all axes for global legend

        for (i, dtype) in enumerate(dtypes)
            row = cld(i, ncols)
            col = i - (row - 1) * ncols
            ax = Axis(
                f[row, col];
                xlabel = "number of events",
                ylabel = "median time elapsed for event generation [ns]",
                yscale = log10,
                xscale = log2,
                title = string(dtype),
            )
            push!(axes, ax)

            bench_dtype_result = bench_result[dtype]

            # Sort batch sizes numerically
            all_bs_sorted = sort!(collect(keys(bench_dtype_result)), by = x -> parse(Int, x))

            for bs in all_bs_sorted
                bench_dtype_bs_result = bench_dtype_result[bs]

                # Sort event numbers numerically
                ns_sorted = sort!(collect(keys(bench_dtype_bs_result)), by = x -> parse(Int, x))
                ns_int = parse.(Int, ns_sorted)
                plot_data = [median(bench_dtype_bs_result[n].time) for n in ns_sorted]

                scatter!(ax, ns_int, plot_data; markersize = 10, label = "batch $bs")
                #lines!(ax, ns_int, plot_data)  # connect points
            end
        end

        # Global legend (below all plots)
        #Legend(f[1:nrows, ncols + 1], axes[end], "batch_size"; orientation = :horizontal)
        Legend(f[nrows + 1, 1:ncols], axes[end], "batch_size"; orientation = :horizontal)

        filename = "$(bench)_$(backend_str).pdf"
        filepath = joinpath(plotpath, filename)
        save(filepath, f)
        @info "Plot saved: $filepath"
    end
end
