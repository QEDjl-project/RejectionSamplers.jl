"""

    get_test_setup(backend::KernelAbstractions.Backend)

Interface function: return test setup for given backend.

"""
function get_test_setup end

abstract type AbstractTestSetup end

struct TestSetup{B <: KernelAbstractions.Backend, VT <: Tuple, T <: Tuple}
    backend::B
    vector_types::VT
    element_types::T
end

function combinations(stp::TestSetup)
    return Iterators.product(stp.vector_types, stp.element_types)
end
