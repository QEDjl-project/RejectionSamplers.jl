# WARNING:
# needs to be included into a scope with the right packages loaded


get_dtypes(result) = collect(keys(result[1]))
get_ns(result, dtype) = collect(keys(result[1][dtype]))
function get_ns(result, dtype, bs)
    all_ns = get_ns(result, dtype)
    ret_ns = []
    for n in all_ns
        if bs in keys(result[1][dtype][n])
            push!(ret_ns, n)
        end
    end
    return ret_ns
end


get_batch_sizes(result, dtype, n) = collect(keys(result[1][dtype][n]))
get_trail(result, dtype, n, bs) = result[1][dtype][n][bs]
get_median_times(result, dtype, n, bs) = median(get_trail(result, dtype, n, bs).times)

function plot_median_times(backend, results)
    dtypes = get_dtypes(results)

    n_dtypes = length(dtypes)
    size_x = 400 * n_dtypes + 200

    f = Figure(size = (size_x, 400))
    for (i, dtype) in enumerate(dtypes)
        @show dtype
        all_ns = get_ns(results, dtype)
        @show all_ns

        ax = Axis(
            f[1, i];
            xlabel = "number of events",
            ylabel = "median time elapsed for event generation [seconds]",
            yscale = log10,
            xscale = log2,
            title = "$(dtype)",
        )

        batch_sizes = get_batch_sizes(results, dtype, all_ns[end])
        @show batch_sizes

        for bs in batch_sizes
            ns_str = get_ns(results, dtype, bs)
            @show ns_str
            ns_int = parse.(Int, ns_str)
            plot_data = get_median_times.(Ref(results), dtype, ns_str, bs)
            scatter!(ax, ns_int, plot_data; markersize = 12, label = "$bs")
        end

        if i == length(dtypes)
            Legend(f[1, length(dtypes)+1], ax, "batch_size")
        end
    end
    filename = "generation_$(backend).pdf"
    @info "plot saved in $(joinpath(plotpath,filename))"
    save(joinpath(plotpath, filename), f)

end
