# main structure adopted from AcceleratedKernels.jl/benchmark/runbenchmarks.jl

using GPUEventGenerators
using GPUEventGenerators.PerturbativeCompton
using QEDcore
using QEDprocesses

import Pkg
using KernelAbstractions
using BenchmarkTools
using StaticArrays

using Random
RNG = Xoshiro(161)
include("utils.jl")

BACKENDS = ["--CUDA", "--oneAPI", "--AMDGPU", "--Metal", "--OpenCL", "--CPU"]
b_opt_idx = in.(ARGS, Ref(BACKENDS))

if !@isdefined(backend_arg)
    backend_arg = if sum(b_opt_idx) == 0
        "--CPU"
    elseif sum(b_opt_idx) == 1
        only(ARGS[b_opt_idx])
    else
        throw(ArgumentError("More than one backend provided. Please retry with only one of $BACKENDS"))
    end
end
backend_arg in BACKENDS || throw(ArgumentError("\"$backend_arg\" is not a valid backend."))

# benchmarks provided by command line argument
other_args = ARGS[.!b_opt_idx]

bench_to_include = isempty(other_args) ? nothing : other_args

# Files to ignore by default. Includes non-benchmark files and
#  backends can add incompatible benchmarks to this list
noinclude = ["utils.jl"]

if backend_arg == "--CUDA"
    @info "Try using CUDA backend."
    "CUDA" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("CUDA")

    using CUDA
    CUDA.versioninfo()

    const BACKEND = CUDABackend()
    const DTYPES = (Float32, Float64)

    macro sb(ex...)
        return quote
            (CUDA.@sync blocking = true $(esc.(ex)...))
        end
    end

    #append!(noinclude, ["sortperm.jl"])

elseif backend_arg == "--oneAPI"
    @info "Try using oneAPI backend."
    "oneAPI" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("oneAPI")

    using oneAPI
    oneAPI.versioninfo()

    const BACKEND = oneAPIBackend()
    const DTYPES = (Float32,)

    macro sb(ex...)
        return quote
            oneAPI.@sync($(esc.(ex)...))
        end
    end
elseif backend_arg == "--AMDGPU"
    @info "Try using AMDGPU backend."
    "AMDGPU" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("AMDGPU")

    using AMDGPU
    AMDGPU.versioninfo()

    const BACKEND = AMDGPUBackend()
    const DTYPES = (Float32, Float64)

    macro sb(ex...)
        return quote
            AMDGPU.@sync($(esc.(ex)...))
        end
    end
elseif backend_arg == "--Metal"
    @info "Try using Metal backend."
    "Metal" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("Metal")

    using Metal
    Metal.versioninfo()

    const BACKEND = MetalBackend()
    const DTYPES = (Float32,)

    macro sb(ex...)
        return quote
            Metal.@sync($(esc.(ex)...))
        end
    end
    #append!(noinclude, ["sort.jl", "sortperm.jl"])

    #=
# TODO: add OpenCL to supported backends
elseif backend_arg == "--OpenCL"
    using OpenCL
    OpenCL.versioninfo()
    const ArrayType = CLArray
    macro sb(ex...) # Not sure how to sync
        quote
            $(esc.(ex)...)
        end
    end
    =#

elseif backend_arg == "--CPU"
    @info "Try using CPU backend."

    "InteractiveUtils" in keys(Pkg.project().dependencies) ? nothing : Pkg.add("InteractiveUtils")
    using InteractiveUtils
    InteractiveUtils.versioninfo()

    const BACKEND = CPU()
    const DTYPES = (Float32, Float64)

    macro sb(ex...)
        return quote
            $(esc.(ex)...)
        end
    end
end


if backend_arg == "--CUDA"
    function reclaim_mem()
        GC.gc(true)
        return CUDA.reclaim()
    end
else
    function reclaim_mem()
        GC.gc(true)
        GC.gc(true)
        return GC.gc(true)
    end
end

include("benchmarks/utils.jl")

# Select benchmarks to run
benches = filter(x -> x ∉ noinclude, Base.readdir("benchmarks"))
if !isempty(other_args)
    benches = filter(x -> any(startswith.(Ref(x), other_args)), benches)
end

SUITE = BenchmarkGroup()
for b in benches
    include(joinpath("benchmarks", b))
end

@info "Preparing benchmarks"
warmup(SUITE; verbose = false)
tune!(SUITE)

reclaim_mem()

@info "Running benchmarks"
results = run(SUITE, verbose = true)

data_filepath = joinpath("data", "bench_$(build_backend_name(backend_arg)).json")
BenchmarkTools.save(data_filepath, median(results))
@info "Save results to $data_filepath"
