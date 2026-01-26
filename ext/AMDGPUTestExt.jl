module AMDGPUTestExt

using RejectionSamplers
using RejectionSamplers.TestUtils
using AMDGPU

@inline function RejectionSamplers.TestUtils.get_test_setup(backend::ROCBackend)
    return TestSetup(backend, (ROCVector,), (Float32, Float64))
end

# register Metal RNGs
RejectionSamplers._rng_strategy(::AMDGPU.rocRAND.RNG) = HostSide()
RejectionSamplers._rng_strategy(::AMDGPU.Device.Philox2x32) = DeviceSide()

# for testing
RejectionSamplers.TestUtils.get_host_rngs(::ROCBackend) = (AMDGPU.gpuarrays_rng(), AMDGPU.rocrand_rng())
RejectionSamplers.TestUtils.get_device_rngs(::ROCBackend) = (AMDGPU.Device.Philox2x32(),)

end
