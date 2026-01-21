"""
    MockArrayBuffer(arr::AbstractArray)

Standard array buffer to test the buffer interface
"""
struct MockArrayBuffer{T, D} <: AbstractBuffer
    data::D

    function MockArrayBuffer(arr::A) where {T, A <: AbstractArray{T}}
        return new{T, A}(arr)
    end
end

function allocate_mock_array_buffer(backend, el_type, size)
    return MockArrayBuffer(
        allocate(backend, el_type, (size,))
    )
end

Adapt.@adapt_structure MockArrayBuffer

KernelAbstractions.get_backend(buf::MockArrayBuffer) = get_backend(buf.data)
Base.length(buf::MockArrayBuffer) = length(buf.data)
Base.eltype(buf::MockArrayBuffer) = eltype(buf.data)
Base.getindex(buf::MockArrayBuffer, idx) = getindex(buf.data, idx)
Base.setindex!(buf::MockArrayBuffer, x, idx) = setindex!(buf.data, x, idx)


struct MockSampleBuffer{Tv, Tw, V, W} <: AbstractSampleBuffer
    values::V
    weights::W

    function MockSampleBuffer(vals::V, ws::W) where {
            Tv, V <: AbstractVector{Tv},
            Tw, W <: AbstractVector{Tw},
        }
        @assert length(vals) == length(ws)
        @assert get_backend(vals) isa typeof(get_backend(ws))

        return new{Tv, Tw, V, W}(vals, w)
    end
end

Adapt.@adapt_structure MockSampleBuffer

KernelAbstractions.get_backend(buf::MockSampleBuffer) = get_backend(buf.values)
Base.length(buf::MockSampleBuffer) = length(buf.values)
Base.eltype(buf::MockSampleBuffer) = eltype(buf.values)
Base.getindex(buf::MockSampleBuffer, idx) = Sample(buf.values[idx], buf.weights[idx])
function Base.setindex!(buf::MockSampleBuffer, sample, idx)
    buf.values[idx] = sample.value
    buf.weights[idx] = sample.weight
    return sample
end
RejectionSamplers.getsample(buf::MockSampleBuffer, idx) = buf[idx]
RejectionSamplers.setsample!(buf::MockSampleBuffer, sample, idx) = (buf[idx] = sample)
