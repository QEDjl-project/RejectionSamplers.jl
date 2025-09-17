function build_backend_name(backend_arg::String)
    return if startswith(backend_arg, "--")
        chop(backend_arg, head = 2, tail = 0)
    else
        throw(
            ArgumentError(
                "Given string is not an command line argument!"
            )
        )
    end
end

get_dtypes(bench, result) = collect(keys(result[1][bench]))
get_ns(bench, result, dtype) = collect(keys(result[1][bench][dtype]))

get_batch_sizes(bench, result, dtype) = collect(keys(result[1][bench][dtype]))
get_ns(bench, result, dtype, bs) = collect(keys(result[1][bench][dtype][bs]))
get_trail(bench, result, dtype, bs, n) = result[1][bench][dtype][bs][n]
get_median_times(bench, result, dtype, bs, n) = median(get_trail(bench, result, dtype, bs, n).times)

function plot_median_times(backend, results)
    dtypes = get_dtypes(results)

    n_dtypes = length(dtypes)
    size_x = 400 * n_dtypes + 200

    f = Figure(size = (size_x, 400))
    for (i, dtype) in enumerate(dtypes)
        @show dtype
        all_bs = get_batch_sizes(results, dtype)
        all_bs_int = sort(parse.(Int, all_bs))
        all_bs_sorted = string.(all_bs_int)
        @show all_bs

        ax = Axis(
            f[1, i];
            xlabel = "number of events",
            ylabel = "median time elapsed for event generation [seconds]",
            yscale = log10,
            xscale = log2,
            title = "$(dtype)",
        )

        for bs in all_bs_sorted
            ns_str = get_ns(results, dtype, bs)
            @show ns_str
            ns_int = sort(parse.(Int, ns_str))
            ns_str_sorted = string.(ns_int)
            @show ns_int
            @show ns_str_sorted
            plot_data = get_median_times.(Ref(results), dtype, bs, ns_str_sorted)
            scatter!(ax, ns_int, plot_data; markersize = 12, label = "$bs")
        end

        if i == length(dtypes)
            Legend(f[1, length(dtypes) + 1], ax, "batch_size")
        end
    end
    filename = "generation_$(backend).pdf"
    @info "plot saved in $(joinpath(plotpath, filename))"
    return save(joinpath(plotpath, filename), f)

end
