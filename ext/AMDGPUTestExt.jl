module AMDGPUTestExt

using RejectionSamplers
using RejectionSamplers.TestUtils
using AMDGPU

@inline function RejectionSamplers.TestUtils.get_test_setup(backend::ROCBackend)
    return TestSetup(backend, (ROCVector,), (Float32, Float64))
end

end
