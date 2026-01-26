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

# register CUDA RNGs
RejectionSamplers._rng_strategy(::CUDA.RNG) = HostSide()
RejectionSamplers._rng_strategy(::CURAND.RNG) = HostSide()
RejectionSamplers._rng_strategy(::CUDA.Philox2x32) = DeviceSide()

# for testing
RejectionSamplers.TestUtils.get_host_rngs(::CUDABackend) = (GPUArrays.default_rng(CuArray), CURAND.RNG(), CUDA.RNG())
RejectionSamplers.TestUtils.get_device_rngs(::CUDABackend) = (CUDA.Philox2x32(),)
end
