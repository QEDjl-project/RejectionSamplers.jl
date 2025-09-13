import Pkg
using KernelAbstractions
using BenchmarkTools
using Random
using QEDcore
using JLD2
using QEDprocesses
using GPUEventGenerators
using GPUEventGenerators.PerturbativeCompton


amdgpu_installed = "AMDGPU" in keys(Pkg.project().dependencies)
@show amdgpu_installed

amdgpu_installed ? nothing : Pkg.add("AMDGPU")

using AMDGPU

if AMDGPU.functional()
    @info "Benchmarking with AMDGPU backend possible!"
else
    throw(ErrorException("AMDGPU backend is not functional (AMDGPU.functional() == false)"))
end

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120.0

include("benchmarks.jl")
include("../params.jl")

HERE = @__DIR__()
DATAPATH = joinpath(HERE, "../data")
@info "datapath: $DATAPATH"

BACKEND = AMDGPUBackend()
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
