"""
    Sample{Ts,Tw} <: AbstractSample{Tw,Ts}

A lightweight container representing a single weighted sample.

# Type Parameters
- `Ts`: Type of the sampled value.
- `Tw`: Type of the sample weight.

# Fields
- `value::Ts`: The sampled value.
- `weight::Tw`: The associated weight of the sample.

This type is intended to be inexpensive to construct and store, and to be used
in bulk (e.g. via `StructArray`) for efficient vectorized or device-backed
sampling workflows.
"""
struct Sample{Ts, Tw} <: AbstractSample{Tw, Ts}
    value::Ts
    weight::Tw
end

@inline value(s::Sample) = s.value
@inline weight(s::Sample) = s.weight

"""
    allocate_samples(backend, valuetype, weighttype, batch_size)

Allocate a batch of samples backed by the given `backend`.

The returned object is a `StructArray{Sample}` whose `value` and `weight`
fields are stored in separate, backend-allocated arrays. This layout enables
efficient bulk operations and compatibility with accelerators such as GPUs.

# Arguments
- `backend`: Allocation backend (e.g. CPU, GPU, or custom allocator).
- `valuetype`: Element type of the sample values.
- `weighttype`: Element type of the sample weights.
- `batch_size::Integer`: Number of samples to allocate.

# Returns
- `StructArray{Sample{valuetype,weighttype}}` of length `batch_size`.

# Notes
This function performs allocation only and does not initialize the contents of
the arrays.
"""
function allocate_samples(backend, valuetype, weighttype, batch_size)
    return StructArray{Sample{valuetype, weighttype}}(
        (
            allocate(backend, valuetype, (batch_size,)), # values
            allocate(backend, weighttype, (batch_size,)), # weights
        )
    )
end
