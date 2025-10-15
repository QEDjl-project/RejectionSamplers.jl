module CUDATestExt

using RejectionSamplers
using RejectionSamplers.TestUtils
using CUDA

@inline function RejectionSamplers.TestUtils.get_test_setup(backend::CUDABackend)
    backends = (
        CUDABackend(),
        CUDABackend(prefer_blocks = true),
        CUDABackend(always_inline = true),
        CUDABackend(prefer_blocks = true, always_inline = true),
    )
    return TestSetup(backends, (CuVector,), (Float16, Float32, Float64))
end

end
