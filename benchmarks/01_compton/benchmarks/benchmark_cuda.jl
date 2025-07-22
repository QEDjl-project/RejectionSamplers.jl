import Pkg
using KernelAbstractions
using BenchmarkTools
using Random
using QEDcore
using JLD2
using QEDprocesses
using GPUEventGenerators
using GPUEventGenerators.PerturbativeCompton


cuda_installed = "CUDA" in keys(Pkg.project().dependencies)

cuda_installed ? nothing : Pkg.add("CUDA")

using CUDA

if CUDA.functional()
    @info "Benchmarking with CUDA backend possible!"
else
    throw(ErrorException("CUDA backend is not functional (CUDA.functional() == false)"))
end

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120.0


"Create random `mu`, `sig`, and `dom` values for given dimension and dtype."
function create_parameters(dtype)
    @info "Create parameters"
    om = dtype(2.0e-3) # 1keV
    @info "om = $om"
    lower = dtype.((om, -1.0, 0.0))
    upper = dtype.((om, 1.0, 2 * pi))
    dom = (lower, upper)

    return dom
end

include("benchmarks.jl")
include("../params.jl")

HERE = @__DIR__()
DATAPATH = joinpath(HERE, "../data")
@info "datapath: $DATAPATH"

BACKEND = CUDABackend()
DTYPES = (Float32, Float64)
@info "backend: $BACKEND"
@info "dtypes: $DTYPES"

const SUITE = BenchmarkGroup()
for dtype in DTYPES
    INSTANCE_SUITE = benchmark_generation(RNG, BACKEND, dtype, MODEL, PSL, NS, BATCH_SIZES)
    SUITE[dtype] = INSTANCE_SUITE
end

@info "Tune the benchmarks"
tune!(SUITE; verbose = true)

@info "Run benchmarks"
result = run(SUITE; verbose = true)


# TODO: is this necessary?
filename = "bench_$(BACKEND).json"
@info "Save results to $filename"
BenchmarkTools.save(joinpath(DATAPATH, filename), result)

filename = "result_$(BACKEND).jld2"
@info "Save results and parameter to $filename"
@save joinpath(DATAPATH, filename) result NS BATCH_SIZES DTYPES
