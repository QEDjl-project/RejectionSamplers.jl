module AMDGPUTestExt

using RejectionSamplers
using RejectionSamplers.TestUtils
using GPUArrays
using AMDGPU


@inline function RejectionSamplers.TestUtils.get_test_setup(backend::ROCBackend)
    return TestSetup(backend, (ROCVector,), (Float32, Float64))
end

# register AMDGPU RNGs
RejectionSamplers._rng_strategy(::AMDGPU.rocRAND.RNG) = HostSide()
#RejectionSamplers._rng_strategy(::AMDGPU.Device.Philox2x32) = DeviceSide()

# for testing
RejectionSamplers.TestUtils.get_host_rngs(::ROCBackend) = (GPUArrays.default_rng(ROCArray),)

RejectionSamplers.TestUtils.get_device_rngs(::ROCBackend) = ()
#RejectionSamplers.TestUtils.get_device_rngs(::ROCBackend) = (AMDGPU.Device.Philox2x32(),)

# NOTE:
# Seems like the AMDGPU.Device.Philox2x32 uses device information upon construction.
# Therefore it hangs if one tries to construct it on host side and pass it to a kernel.
# So, currently, we can not support device-side rngs (in our understanding) for the
# ROCBackend.

end
