"""
AbstractBuffer

Interface functions (maybe the array interface?)
- `KernelAbstractions.get_backend(buf)`
    - `Base.length(buf)`
    - `Base.eltype(buf)`
    - `getindex(buf,idx)`
    - `setindex!(buf,idx,x)`
"""
abstract type AbstractBuffer end

"""
AbstractSampleBuffer

Interface functions

from AbstractBuffer:
- `KernelAbstractions.get_backend(buf)`
    - `Base.length(buf)`
    - `Base.eltype(buf)` -> Sample{value_type(buf),weight_type(buf)}
    - `Base.getindex(buf,idx)`
    - `Bsae.setindex!(buf,x,idx)`

    additionally:
    - `value_type(buf)`
    - `weight_type(buf)`
    - `getsample(buf,idx)`
    - `setsample!(buf,sample, idx)`
"""
abstract type AbstractSampleBuffer <: AbstractBuffer end

"""
    value_type(buf)
"""
function value_type end

"""
    weight_type(buf)
"""
function weight_type end

"""
    getsample(buf,idx)
"""
function getsample end

"""
    setsample!(buf, sample, idx)

Optional
"""
function setsample! end

"""
    setsamples!(buf, samples)

Optional, bulk alternative for setsample!
"""
function setsamples! end


### getter

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

function getsamples(buf::AbstractSampleBuffer)
    out = allocate(get_backend(buf), eltype(buf), length(buf))
    _getsamples!(buf, out)
    return out
end

getvalue(buf::AbstractSampleBuffer, idx) = getsample(buf, idx).value
function getvalues(buf::AbstractSampleBuffer)
    return getfield.(getsamples(buf), :value)
end

getweight(buf::AbstractSampleBuffer, idx) = getsample(buf, idx).weight
function getweights(buf::AbstractSampleBuffer)
    return getfield.(getsamples(buf), :weight)
end


### setter

function _setsamples!(buf, samples)
    broadcast(Base.Fix1(setsample!, buf), samples, 1:length(buf))
    return setsample!.(buf, samples, 1:length(buf))
end

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
    get_backend(samples) isa typeof(get_backend(buf)) || throw(
        ArgumentError(
            "buffer and samples vector must have compatible backends"
        )
    )
    return _setsamples!(buf, samples)
end

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
