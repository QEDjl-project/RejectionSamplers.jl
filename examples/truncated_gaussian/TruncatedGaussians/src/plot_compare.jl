function plot_compare(samples, dist; kwargs...)
    P = histogram(
        samples;
        label = "samples",
        xlabel = "x",
        ylabel = "normalized event count",
        nbins = 100,
        normalize = :pdf,
        opacity = 0.5,
        kwargs...,
    )

    tot_weight, _ = quadgk(x -> GPUEventGenerators._compute(dist, x), endpoints(dist)...)
    plot!(
        P,
        range(endpoints(dist)...; length = 100),
        x -> GPUEventGenerators._compute(dist, x) / tot_weight;
        label = "normalized target dist.",
        line = (2, :black, :dash),
        alpha = 0.5,
    )

    return P
end
