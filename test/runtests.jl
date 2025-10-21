using Pkg
using Test
using Random
using GPUArrays
using KernelAbstractions
using StaticArrays

using RejectionSamplers

include("../examples/truncated_gaussian/truncated_gaussian.jl")

using .TruncatedGaussians
using RejectionSamplers.TestUtils
using RejectionSamplers.Mocks

SETUPS = TestSetup[]

# check if we test with CPU
cpu_tests = tryparse(Bool, get(ENV, "TEST_CPU", "1"))
if cpu_tests
    push!(SETUPS, get_test_setup(CPU()))
    @info "Testing with CPU backend"
else
    @info "CPU tests skipped."
end

# check if we test with Metal
metal_tests = tryparse(Bool, get(ENV, "TEST_METAL", "0"))
metal_installed = "Metal" in keys(Pkg.project().dependencies)
if metal_tests

    metal_installed ? nothing : Pkg.add("Metal")

    using Metal

    if Metal.functional()
        push!(SETUPS, get_test_setup(MetalBackend()))
        @info "Testing with Metal backend"
    else
        @error "Metal backend is not functional (Metal.functional() == false)"
        @test false
    end

else
    metal_installed ? @warn("Metal is installed, but tests skipped.") :
        @info("Metal tests skipped.")
end

# check if we test with CUDA
cuda_tests = tryparse(Bool, get(ENV, "TEST_CUDA", "0"))
cuda_installed = "CUDA" in keys(Pkg.project().dependencies)
if cuda_tests

    cuda_installed ? nothing : Pkg.add("CUDA")

    using CUDA

    if CUDA.functional()
        push!(SETUPS, get_test_setup(CUDABackend()))
        @info "Testing with CUDA backend"
    else
        @error "CUDA backend is not functional (CUDA.functional() == false)"
        @test false
    end

else
    cuda_installed ? @warn("CUDA is installed, but tests skipped.") :
        @info("CUDA tests skipped.")
end

# check if we test with oneAPI
oneapi_tests = tryparse(Bool, get(ENV, "TEST_ONEAPI", "0"))
oneapi_installed = "oneAPI" in keys(Pkg.project().dependencies)
if oneapi_tests

    oneapi_installed ? nothing : Pkg.add("oneAPI")

    using oneAPI

    if oneAPI.functional()
        push!(SETUPS, get_test_setup(oneAPIBackend()))
        @info "Testing with oneAPI backend"
    else
        @error "oneAPI backend is not functional (oneAPI.functional() == false)"
        @test false
    end

else
    oneapi_installed ? @warn("oneAPI is installed, but tests skipped.") :
        @info("oneAPI tests skipped.")
end

# check if we test with AMDGPU
amdgpu_tests = tryparse(Bool, get(ENV, "TEST_AMDGPU", "0"))
amdgpu_installed = "AMDGPU" in keys(Pkg.project().dependencies)
if amdgpu_tests

    amdgpu_installed ? nothing : Pkg.add("AMDGPU")

    using AMDGPU

    if AMDGPU.functional()
        push!(SETUPS, get_test_setup(ROCBackend()))
        @info "Testing with AMDGPU backend"
    else
        @error "AMDGPU backend is not functional (AMDGPU.functional() == false)"
        @test false
    end

else
    amdgpu_installed ? @warn("AMDGPU is installed, but tests skipped.") :
        @info("AMDGPU tests skipped.")
end

# from here on, we cannot use safe test sets or we would unload the GPU libraries again
if isempty(SETUPS)
    @info """No backends are enabled, skipping tests...
    To test a backend, please use 'TEST_<BACKEND> = 1 julia ...' for one of BACKEND=[CPU, CUDA, AMDGPU, METAL, ONEAPI]"""
    return nothing
end


include("testsuite.jl")

for stp in SETUPS
    @testset "$backend $vec_type $el_type" for (backend, vec_type, el_type) in combinations(stp)
        testsuite_run(backend, vec_type, el_type)
    end
end
