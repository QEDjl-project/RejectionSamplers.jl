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
    - `Base.eltype(buf)` -> Sample{valtype(buf),wtype(buf)}
    - `Base.getindex(buf,idx)`
    - `Bsae.setindex!(buf,idx,x)`

    additionally:
    - `value_type(buf)`
    - `weight_type(buf)`
    - `getsample(buf,idx)`
    - `setsample!(buf,idx)`
"""
abstract type AbstractSampleBuffer <: AbstractBuffer end

"""
    valtype(buf)
"""
function valtype end

"""
    wtype(buf)
"""
function wtype end

"""
    getsample(buf,idx)
"""
function getsample end

"""
    setsample!(buf, idx, sample)

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
    return setsample!.(buf, 1:length(buf), samples)
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

@inline function setvalue!(buf, idx, value)
    return setsample!(
        buf,
        idx,
        Sample(
            value,
            getweight(buf, idx)
        )
    )
end

function _setvalues!(buf, values)
    return setvalue!.(buf, 1:length(buf), values)
end

function setvalues!(buf, values)
    length(buf) == length(values) || throw(
        ArgumentError(
            "buffer and values vector must have the same length"
        )
    )

    valtype(buf) == eltype(values) ||  throw(
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

@inline function setweight!(buf, idx, weight)
    return setsample!(
        buf,
        idx,
        Sample(
            getvalue(buf, idx),
            weight
        )
    )
end

function _setweights!(buf, weights)
    return setweight!.(buf, 1:length(buf), weights)
end

function setweights!(buf, weights)
    length(buf) == length(weights) || throw(
        ArgumentError(
            "buffer and weights vector must have the same length"
        )
    )

    valtype(buf) == eltype(weights) ||  throw(
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
