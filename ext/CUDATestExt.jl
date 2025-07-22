module CUDATestExt

using GPUEventGenerators
using GPUEventGenerators.TestUtils
using CUDA

@inline function GPUEventGenerators.TestUtils.get_test_setup(backend::CUDABackend)
    return TestSetup(backend, (CuVector,), (Float16, Float32, Float64))
end

end
