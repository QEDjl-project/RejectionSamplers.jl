module oneAPITestExt

using RejectionSamplers
using RejectionSamplers.TestUtils
using oneAPI

@inline function RejectionSamplers.TestUtils.get_test_setup(backend::oneAPIBackend)

    # check if f64 is supported
    if oneL0.module_properties(oneAPI.device()).fp64flags &
            oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64
        element_types = (Float32, Float64)
    else
        element_types = (Float32,)
    end

    return TestSetup(backend, (oneVector,), element_types)
end

end
