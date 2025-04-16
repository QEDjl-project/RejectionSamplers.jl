using Test

VECTOR_TYPES = Vector{Type}()
FLOAT_TYPES = Dict{Type,Vector{Type}}()

# check if we test with CPU
cpu_tests = tryparse(Bool, get(ENV, "TEST_CPU", "1"))
if cpu_tests
    push!(VECTOR_TYPES, Vector)
    FLOAT_TYPES[Vector] = [Float16, Float32, Float64]

    @info "Testing with CPU backend"
end

# check if we test with AMDGPU
amdgpu_tests = tryparse(Bool, get(ENV, "TEST_AMDGPU", "0"))
if amdgpu_tests
    try
        using Pkg
        Pkg.add("AMDGPU")

        using AMDGPU
        AMDGPU.functional() || throw(
            "trying to test with AMDGPU.jl but it is not functional (AMDGPU.functional() == false)",
        )
        push!(VECTOR_TYPES, ROCVector)
        FLOAT_TYPES[ROCVector] = [Float32, Float64]

        @info "Testing with AMDGPU.jl"
    catch e
        @error "failed to run GPU tests, make sure the required libraries are installed\n$(e)"
        @test false
    end
end

# check if we test with CUDA
cuda_tests = tryparse(Bool, get(ENV, "TEST_CUDA", "0"))
if cuda_tests
    try
        using Pkg
        Pkg.add("CUDA")

        using CUDA
        CUDA.functional() || throw(
            "trying to test with CUDA.jl but it is not functional (CUDA.functional() == false)",
        )
        push!(VECTOR_TYPES, CuVector)
        FLOAT_TYPES[CuVector] = [Float32, Float64]

        @info "Testing with CUDA.jl"
    catch e
        @error "failed to run GPU tests, make sure the required libraries are installed\n$(e)"
        @test false
    end
end

# check if we test with oneAPI
oneapi_tests = tryparse(Bool, get(ENV, "TEST_ONEAPI", "0"))
if oneapi_tests
    try
        using Pkg
        Pkg.add("oneAPI")

        using oneAPI
        oneAPI.functional() || throw(
            "trying to test with oneAPI.jl but it is not functional (oneAPI.functional() == false)",
        )
        push!(VECTOR_TYPES, oneVector)
        FLOAT_TYPES[oneVector] = [Float32]
        if oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 ==
           oneL0.ZE_DEVICE_MODULE_FLAG_FP64
            # This checks whether the Intel GPU supports Float64, see oneAPI Readme
            push!(FLOAT_TYPES[oneVector], Float64)
        end

        @info "Testing with oneAPI.jl"
    catch e
        @error "failed to run GPU tests, make sure the required libraries are installed\n$(e)"
        @test false
    end
end

# check if we test with Metal
metal_tests = tryparse(Bool, get(ENV, "TEST_METAL", "0"))
if metal_tests
    try
        using Pkg
        Pkg.add("Metal")

        using Metal
        Metal.functional() || throw(
            "trying to test with Metal.jl but it is not functional (Metal.functional() == false)",
        )
        push!(VECTOR_TYPES, MtlVector)
        FLOAT_TYPES[MtlVector] = [Float32]

        @info "Testing with Metal.jl"
    catch e
        @error "failed to run GPU tests, make sure the required libraries are installed\n$(e)"
        @test false
    end
end

# from here on, we cannot use safe test sets or we would unload the GPU libraries again
if isempty(VECTOR_TYPES)
    @info """No backends are enabled, skipping tests...
    To test a backend, please use 'TEST_<BACKEND> = 1 julia ...' for one of BACKEND=[CPU, CUDA, AMDGPU, METAL, ONEAPI]"""
    return nothing
end

include("filter_scan.jl")
