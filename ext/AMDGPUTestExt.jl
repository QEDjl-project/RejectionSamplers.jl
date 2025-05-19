
module AMDGPUTestExt

using GPUEventGenerators
using GPUEventGenerators.TestUtils
using AMDGPU

@inline function GPUEventGenerators.TestUtils.get_test_setup(backend::ROCBackend)
    TestSetup(backend, (ROCVector,), (Float32, Float64))
end

end
