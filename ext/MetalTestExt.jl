module MetalTestExt

using RejectionSamplers
using RejectionSamplers.TestUtils
using Metal

@inline function RejectionSamplers.TestUtils.get_test_setup(backend::MetalBackend)
    return TestSetup(backend, (MtlVector,), (Float16, Float32))
end


end
