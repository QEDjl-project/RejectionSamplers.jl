module TestUtils

export TestSetup, combinations, get_test_setup

using KernelAbstractions
using Random


"""

    get_test_setup(backend::KernelAbstractions.Backend)

Interface function: return test setup for given backend.

"""
function get_test_setup end

"""
    get_host_rngs(backend::KernelAbstractions.Backend)

Interface function: return a tuple of compatible host-side random number generators
"""
function get_host_rngs end

"""
    get_device_rngs(backend::KernelAbstractions.Backend)

Interface function: return a tuple of compatible device-side random number generators
"""
function get_device_rngs end

abstract type AbstractTestSetup end

struct TestSetup{VB <: Tuple, VT <: Tuple, T <: Tuple}
    backend::VB
    vector_types::VT
    element_types::T
end

function TestSetup(backend::Backend, vector_types::Tuple, element_types::Tuple)
    return TestSetup((backend,), vector_types, element_types)
end


function combinations(stp::TestSetup)
    return Iterators.product(stp.backend, stp.vector_types, stp.element_types)
end

# CPU test setup

function get_test_setup(backend::CPU)
    backends = (
        CPU(),
        CPU(static = true),
    )
    return TestSetup(backends, (Vector,), (Float16, Float32, Float64))
end

get_host_rngs(::CPU) = (Xoshiro(),)
get_device_rngs(::CPU) = ()

end
