using KernelAbstractions
using BenchmarkTools
using CairoMakie

include("utils.jl")

BACKEND = CPU()

here = @__DIR__()
plotpath = joinpath(here, "../plots/")
datapath = joinpath(here, "../data/")

filename = "bench_$(BACKEND).json"
results_cpu = BenchmarkTools.load(joinpath(datapath, filename))

### median time elapsed
plot_median_times(BACKEND, results_cpu)

# TODO: add more plots
# - memory measurements
# - gc usage
# - compile-time?
