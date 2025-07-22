using KernelAbstractions
using BenchmarkTools
using Random
using GPUEventGenerators
using GPUEventGenerators.PerturbativeCompton

using CairoMakie
using PairPlots

const RNG = Xoshiro(1234)
const PLOTDIR = "plots"

"Create random `mu`, `sig`, and `dom` values for given dimension and dtype."
function create_parameters(dtype)
    @info "Create parameters"
    om = dtype(2.0e-3) # 1keV
    @info "om = $om"
    #lower = ntuple(i -> mu[i] - dtype(3.0) * sig[i], dim)
    lower = dtype.((-1.0, 0.0))
    upper = dtype.((1.0, 2 * pi))
    dom = (lower, upper)

    return om, dom
end

function plot_corner(
        samples,
        backend,
        dtype,
        filename = "corner_plot_$(backend)_$(dtype).png",
    )
    @info "Generating corner plot"
    data = ntuple(2) do i
        getindex.(samples, i)
    end
    labels = (:cos_theta, :phi)
    nt = NamedTuple{labels}(data)
    fig = pairplot(nt)

    return save(joinpath(PLOTDIR, filename), fig)
end

function plot_weights(weights, backend, dtype, filename = "weights_$(backend)_$(dtype).png")

    f = Figure()
    ax = Axis(f[1, 1], xlabel = "weights", ylabel = "event count", yscale = log10)

    hist!(ax, weights, color = :orange, bins = 100)

    return save(joinpath(PLOTDIR, filename), f)
end

function run_example(
        backend,
        dtype;
        plotting = true,
        partial_unweighting = true,
        N = Int(2^20),
        batch_size = 2^20,
    )
    @info "Running example with backend=$backend, dtype=$dtype"

    om, dom = create_parameters(dtype)

    dist = KleinNishinaDistribution(om)
    proposal = UniformMultivariateProposal(dom...)

    scale =
        partial_unweighting ? (@info "Partial unweighting enabled"; dtype(0.9)) : one(dtype)
    max_value = maximum_value(dist) * scale

    # Generate events
    @info "Generate N = $N events with batch_size=$batch_size"
    data = GPUEventGenerators.generate_events(
        dist,
        proposal,
        max_value,
        N,
        batch_size,
        backend,
        dtype,
        NTuple{2, dtype},
    )


    # Plotting
    if plotting
        plot_corner(data[1], backend, dtype)
        plot_weights(data[2], backend, dtype)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Run on CPU
    #run_example(CPU(), Float64; dim = 2)

    # Run on GPU
    using Metal
    run_example(MetalBackend(), Float32; N = Int(2^22))
end
