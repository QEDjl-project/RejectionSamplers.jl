"""
    AbstractBuffer

Abstract supertype for vector-like buffers that can be used as targets for
CPU- or GPU-based kernels.

An `AbstractBuffer` represents a one-dimensional, indexable container with
a well-defined execution backend. Concrete implementations are expected
to behave similarly to arrays, but may wrap backend-specific storage
(e.g. GPU memory).

### Required interface

- `KernelAbstractions.get_backend(buf)`
- `Base.length(buf)`
- `Base.eltype(buf)`
- `Base.getindex(buf, idx)`
- `Base.setindex!(buf, x, idx)`
"""
abstract type AbstractBuffer end

"""
    AbstractSampleBuffer <: AbstractBuffer

Abstract buffer type specialized for storing `Sample` objects consisting
of a value and an associated weight.

An `AbstractSampleBuffer` extends `AbstractBuffer` by providing accessors
and mutators at the level of whole samples, as well as at the level of
values and weights individually.

### Required interface

From `AbstractBuffer`:
- `KernelAbstractions.get_backend(buf)`
- `Base.length(buf)`
- `Base.eltype(buf)`
  Must return `Sample{value_type(buf), weight_type(buf)}`
- `Base.getindex(buf, idx)`
- `Base.setindex!(buf, x, idx)`

Additional sample-specific interface:
- `value_type(buf)`
- `weight_type(buf)`
- `getsample(buf, idx)`
- `setsample!(buf, sample, idx)` (optional)

The implementation of `setsample!` is optional, because if the buffer is static, or if there are other
way implemented to update the samples, this one is not needed.
"""
abstract type AbstractSampleBuffer <: AbstractBuffer end

"""
    value_type(buf)

Return the element type of the `value` of the samples contained in `buf`.
"""
function value_type end

"""
    weight_type(buf)

Return the element type of the `weight` of the samples contained in `buf`.
"""
function weight_type end

"""
    getsample(buf, idx)

Return the `Sample` stored at index `idx` in `buf`.
"""
function getsample end

"""
    setsample!(buf, sample, idx)

Store `sample` at index `idx` in `buf`.

This method is optional; bulk setters such as `setsamples!` may be
implemented instead. If both are provided, `setsample!` should represent
the canonical element-wise operation.
"""
function setsample! end

"""
    setsamples!(buf, samples)

Store all elements from `samples` into `buf`.

This is an optional bulk alternative to repeated calls to `setsample!`.
Implementations may override this method to provide a more efficient
backend-specific implementation.
"""
function setsamples! end


### getter api

"""
    getsamples!(buf, out)

Copy all samples stored in `buf` into the preallocated vector `out`.

The buffer and output vector must have the same length, element type,
and compatible execution backends.
"""
function getsamples!(buf::AbstractSampleBuffer, out::AbstractVector)
    length(buf) == length(out) || throw(
        ArgumentError(
            "buffer and out vector must have the same length"
        )
    )

    eltype(buf) == eltype(out) ||  throw(
        ArgumentError(
            "buffer and out vector must have the same element type"
        )
    )
    get_backend(out) isa typeof(get_backend(buf)) || throw(
        ArgumentError(
            "buffer and out vector must have compatible backends"
        )
    )
    return _getsamples!(buf, out)
end

function _getsamples!(buf::AbstractSampleBuffer, out::AbstractVector)
    map!(Base.Fix1(getsample, buf), out, 1:length(buf))
    return out
end

"""
    getsamples(buf)

Return a newly allocated vector containing all samples stored in `buf`.

The returned vector is allocated on the same backend as `buf`.
"""
function getsamples(buf::AbstractSampleBuffer)
    out = allocate(get_backend(buf), eltype(buf), length(buf))
    _getsamples!(buf, out)
    return out
end

"""
    getvalue(buf, idx)

Return the `value` field of the sample stored at index `idx`.
"""
getvalue(buf::AbstractSampleBuffer, idx) = getsample(buf, idx).value

"""
    getvalues(buf)

Return a vector containing the `value` field of all samples stored in `buf`.
"""
function getvalues(buf::AbstractSampleBuffer)
    return getfield.(getsamples(buf), :value)
end

"""
    getweight(buf, idx)

Return the `weight` field of the sample stored at index `idx`.
"""
getweight(buf::AbstractSampleBuffer, idx) = getsample(buf, idx).weight

"""
    getweights(buf)

Return a vector containing the `weight` field of all samples stored in `buf`.
"""
function getweights(buf::AbstractSampleBuffer)
    return getfield.(getsamples(buf), :weight)
end


### setter

function _setsamples!(buf, samples)
    broadcast(Base.Fix1(setsample!, buf), samples, 1:length(buf))
    return setsample!.(buf, samples, 1:length(buf))
end

"""
    setsamples!(buf, samples)

Store all samples from `samples` into `buf`.

The buffer and input vector must have the same length, element type,
and compatible execution backends.
"""
function setsamples!(buf, samples)
    length(buf) == length(samples) || throw(
        ArgumentError(
            "buffer and samples vector must have the same length"
        )
    )

    eltype(buf) == eltype(samples) ||  throw(
        ArgumentError(
            "buffer and samples vector must have the same element type"
        )
    )
    return _setsamples!(buf, samples)
end

"""
    setvalue!(buf, value, idx)

Update only the `value` of the sample at index `idx`, leaving the
associated weight unchanged.
"""
@inline function setvalue!(buf, value, idx)
    return setsample!(
        buf,
        Sample(
            value,
            getweight(buf, idx)
        ),
        idx,
    )
end

function _setvalues!(buf, values)
    return setvalue!.(buf, values, 1:length(buf))
end

"""
    setvalues!(buf, values)

Update the `value` field of all samples stored in `buf`, leaving all
weights unchanged.

The buffer and input vector must have the same length, value type,
and compatible execution backends.
"""
function setvalues!(buf, values)
    length(buf) == length(values) || throw(
        ArgumentError(
            "buffer and values vector must have the same length"
        )
    )

    value_type(buf) == eltype(values) ||  throw(
        ArgumentError(
            "buffer and values vector must have the same element type"
        )
    )
    get_backend(values) isa typeof(get_backend(buf)) || throw(
        ArgumentError(
            "buffer and values vector must have compatible backends"
        )
    )
    return _setvalues!(buf, values)
end

"""
    setweight!(buf, weight, idx)

Update only the `weight` of the sample at index `idx`, leaving the
associated value unchanged.
"""
@inline function setweight!(buf, weight, idx)
    return setsample!(
        buf,
        Sample(
            getvalue(buf, idx),
            weight
        ),
        idx,
    )
end

function _setweights!(buf, weights)
    return setweight!.(buf, weights, 1:length(buf))
end

"""
    setweights!(buf, weights)

Update the `weight` of all samples stored in `buf`, leaving all
values unchanged.

The buffer and input vector must have the same length, weight type,
and compatible execution backends.
"""
function setweights!(buf, weights)
    length(buf) == length(weights) || throw(
        ArgumentError(
            "buffer and weights vector must have the same length"
        )
    )

    value_type(buf) == eltype(weights) ||  throw(
        ArgumentError(
            "buffer and weights vector must have the same element type"
        )
    )
    get_backend(weights) isa typeof(get_backend(buf)) || throw(
        ArgumentError(
            "buffer and weights vector must have compatible backends"
        )
    )
    return _setweights!(buf, weigths)
end
