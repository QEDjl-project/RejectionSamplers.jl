
# some imports

using KernelAbstractions
using BenchmarkTools
using Random
using Plots
const RNG = MersenneTwister(1234)

using GPUEventGenerators

# import example
using GPUEventGenerators.TruncatedGaussians

using StatsPlots


const PLOTDIR = "plots"

function run_example(
    backend,
    dtype;
    plotting = true,
    partial_unweighting = true,
    N = Int(2^20),
    batch_size = 2^20,
)

    @info "run example for $backend with $dtype"
    # setup

    @info "init example dist"
    mu = 5 * rand(RNG, dtype)                  # central value
    sig = rand(RNG, dtype)                 # variance
    dom = dtype.((mu - 0.5 * sig, mu + 3.0 * sig))   # support/domain

    ## example distribution to be sampled
    dist = TruncatedGaussian1D(mu, sig, dom)
    proposal = UniformProposal(dom)

    @info "find maximum"
    ## max value (bit smaller to enable partial unweighting)
    scale =
        partial_unweighting ? (@info "partial unweighting enabled"; dtype(0.9)) : one(dtype)
    max_value = maximum_value(dist) * scale

    ## number of samples to be generated

    @info "generate events"
    data = GPUEventGenerators.generate_events(
        dist,
        proposal,
        max_value,
        N,
        batch_size,
        backend,
        dtype,
    )

    if plotting
        @info "plot samples"
        ## plotting: histogram samples vs target_dist
        P = TruncatedGaussians.plot_compare(
            data[1],
            dist;
            title = "samples ($backend, $dtype)",
        )

        ## save plot
        plotpath_result = joinpath(PLOTDIR, "example_samples_$(backend)_$(dtype).pdf")
        savefig(P, plotpath_result)
        @info "saved in $plotpath_result"

        @info "plot weights"
        ## plotting: weights
        P_weights = histogram(
            data[2],
            xlab = "weights",
            ylab = "count",
            lab = "weights",
            title = "samples ($backend, $dtype)",
        )

        ## save plot
        plotpath_weights = joinpath(PLOTDIR, "example_weights_$(backend)_$(dtype).pdf")
        savefig(P_weights, plotpath_weights)
        @info "saved in $plotpath_weights"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_example(CPU(), Float16)
    run_example(CPU(), Float32)
    run_example(CPU(), Float64)

    # coment this out, if you would like to run the example on different backends

    using Metal
    run_example(MetalBackend(), Float16)
    run_example(MetalBackend(), Float32)

    #using CUDA
    #run_example(CUDABackend(),Float16)
    #run_example(CUDABackend(),Float32)
    #run_example(CUDABackend(),Float64)
end
