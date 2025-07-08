module PlotEx

using StatsPlots
using QuadGK
using GPUEventGenerators

function GPUEventGenerators.plot_compare(
    samples::AbstractVector,
    dist::GPUEventGenerators.AbstractUnivariatTargetDistribution;
    kwargs...,
)
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

    tot_weight, _ = quadgk(x -> GPUEventGenerators._compute(dist, x), extrema(dist)...)
    plot!(
        P,
        range(extrema(dist)...; length = 100),
        x -> GPUEventGenerators._compute(dist, x) / tot_weight;
        label = "normalized target dist.",
        line = (2, :black, :dash),
        alpha = 0.5,
    )

    return P
end

end
