using KernelAbstractions
using BenchmarkTools
using Random
using RejectionSamplers

include("truncated_gaussian.jl")
using .TruncatedGaussians

using CairoMakie
using PairPlots
using DataFrames

const RNG = MersenneTwister(1234)
const PLOTDIR = "plots"

"Create random `mu`, `sig`, and `dom` values for given dimension and dtype."
function create_parameters(dim::Int, dtype)
    @info "Create parameters"
    mu = ntuple(_ -> rand(RNG, dtype) * 4 .- dtype(2.0), dim)                # values roughly in [-2, 2]
    @info "mu = $mu"
    sig = ntuple(_ -> rand(RNG, dtype) * 2 .+ dtype(0.5), dim)              # values roughly in [0.5, 2.5]
    @info "sig = $sig"

    #lower = ntuple(i -> mu[i] - dtype(3.0) * sig[i], dim)
    lower = ntuple(i -> -dtype(3.0) * sig[i], dim)
    upper = ntuple(i -> mu[i] + dtype(3.0) * sig[i], dim)
    dom = (lower, upper)

    return mu, sig, dom
end

function plot_corner(
        samples,
        backend,
        dim,
        dtype,
        filename = "corner_plot_$(backend)_$(dtype)_$(dim)D.png",
    )
    @info "Generating corner plot"
    data = ntuple(dim) do i
        getindex.(samples, i)
    end
    labels = Symbol.(ntuple(i -> "x_$i", dim))
    nt = NamedTuple{labels}(data)
    fig = PairPlots.pairplot(nt)

    return save(joinpath(PLOTDIR, filename), fig)
end

function plot_weights(
        weights,
        backend,
        dim,
        dtype,
        filename = "weights_$(backend)_$(dtype)_$(dim)D.png",
    )

    f = Figure()
    ax = Axis(f[1, 1], xlabel = "weights", ylabel = "event count")

    hist!(ax, weights, color = :orange, bins = 100)

    return save(joinpath(PLOTDIR, filename), f)
end

function run_example(
        backend,
        dtype;
        plotting = true,
        partial_unweighting = true,
        dim = 2,
        N = Int(2^20),
        batch_size = 2^20,
    )
    @info "Running example with backend=$backend, dtype=$dtype, dim=$dim"

    mu, sig, dom = create_parameters(dim, dtype)

    dist = TruncatedGaussian(mu, sig, dom...)
    proposal = UniformMultivariateProposal(dom...)


    scale =
        partial_unweighting ? (@info "Partial unweighting enabled"; dtype(0.9)) : one(dtype)
    max_value = maximum_value(dist) * scale

    # Generate events
    @info "Generate N = $N events with batch_size=$batch_size"
    data = RejectionSamplers.generate_events(
        dist,
        proposal,
        max_value,
        N,
        batch_size,
        backend,
        dtype,
        NTuple{dim, dtype},
    )


    # Plotting
    if plotting

        plot_corner(data[1], backend, dim, dtype)
        plot_weights(data[2], backend, dim, dtype)

    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Run on CPU
    #run_example(CPU(), Float64; dim = 2)

    # Run on GPU
    using Metal
    run_example(MetalBackend(), Float32; dim = 4, N = Int(2^22))
end
