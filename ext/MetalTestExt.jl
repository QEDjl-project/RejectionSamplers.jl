module MetalTestExt

using GPUEventGenerators
using GPUEventGenerators.TestUtils
using Metal

@inline function GPUEventGenerators.TestUtils.get_test_setup(backend::MetalBackend)
    return TestSetup(backend, (MtlVector,), (Float16, Float32))
end


end
