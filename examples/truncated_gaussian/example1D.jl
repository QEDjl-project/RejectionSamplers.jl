
# some imports

using KernelAbstractions
using BenchmarkTools
using Random
using QuadGK

const RNG = MersenneTwister(1234)

using GPUEventGenerators
using GPUEventGenerators.TruncatedGaussians

using CairoMakie

const PLOTDIR = "plots"

"Create random `mu`, `sig`, and `dom` values for given dimension and dtype."
function create_parameters(dtype)
    @info "Create parameters"
    mu = rand(RNG, dtype) * 4 - dtype(2.0)                         # values roughly in [-2, 2]
    @info "mu = $mu"
    sig = rand(RNG, dtype) * 2 .+ dtype(0.5)                        # values roughly in [0.5, 2.5]
    @info "sig = $sig"

    lower = -dtype(3.0) * sig
    upper = mu + dtype(3.0) * sig
    dom = (lower, upper)

    return mu, sig, dom
end

function plot_samples(
    dist,
    samples,
    backend,
    dtype,
    filename = "compare_$(backend)_$(dtype)_1D.png",
)
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "x", ylabel = "normalized event count")
    hist!(ax, samples, normalization = :pdf, bins = 100, color = :orange)

    tot_weight, _ = quadgk(x -> GPUEventGenerators._compute(dist, x), extrema(dist)...)
    x = range(extrema(dist)..., 100)
    vals = @. GPUEventGenerators._compute(dist, x) / tot_weight

    lines!(ax, x, vals, linestyle = :dash, color = :black)

    save(joinpath(PLOTDIR, filename), f)
end

function plot_weights(
    weights,
    backend,
    dtype,
    filename = "weights_$(backend)_$(dtype)_1D.png",
)

    f = Figure()
    ax = Axis(f[1, 1], xlabel = "weights", ylabel = "event count")

    hist!(ax, weights, color = :orange, bins = 100)

    save(joinpath(PLOTDIR, filename), f)
end


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

    mu, sig, dom = create_parameters(dtype)

    ## example distribution to be sampled
    dist = TruncatedGaussian1D(mu, sig, dom)
    proposal = UniformUnivariateProposal(dom)

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
        plot_samples(dist, data[1], backend, dtype)
        plot_weights(data[2], backend, dtype)

    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    #run_example(CPU(), Float16)
    #run_example(CPU(), Float32)
    run_example(CPU(), Float64)

    # coment this out, if you would like to run the example on different backends

    #using Metal
    #run_example(MetalBackend(), Float16; N = Int(2^22), partial_unweighting = false)
    #run_example(MetalBackend(), Float32; N = Int(2^22), partial_unweighting = false)

    #using CUDA
    #run_example(CUDABackend(),Float16)
    #run_example(CUDABackend(),Float32)
    #run_example(CUDABackend(),Float64)
end
