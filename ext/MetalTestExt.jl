module MetalTestExt

using RejectionSamplers
using RejectionSamplers.TestUtils
using GPUArrays
using Metal

@inline function RejectionSamplers.TestUtils.get_test_setup(backend::MetalBackend)
    return TestSetup(backend, (MtlVector,), (Float16, Float32))
end

# register Metal RNGs
RejectionSamplers._rng_strategy(::Metal.MPS.RNG) = HostSide()
RejectionSamplers._rng_strategy(::Metal.Philox2x32) = DeviceSide()

# for testing
RejectionSamplers.TestUtils.get_host_rngs(::MetalBackend) = (GPUArrays.default_rng(MtlVector), Metal.MPS.RNG())
RejectionSamplers.TestUtils.get_device_rngs(::MetalBackend) = (Metal.Philox2x32(),)

end
