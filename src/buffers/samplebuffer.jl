"""
    SampleBuffer{Tv, Tw, S} <: AbstractSampleBuffer

Concrete implementation of `AbstractSampleBuffer` that stores weighted
samples in a structure-of-arrays layout.

`SampleBuffer` represents a one-dimensional buffer of `Sample{Tv,Tw}`
elements, where values and weights are stored in separate contiguous
arrays via a `StructVector`. This layout enables efficient
field-wise access and mutation, and is suitable for both CPU and GPU
backends.

### Type parameters
- `Tv`: value type of the samples
- `Tw`: weight type of the samples
- `S`: underlying storage type (typically `StructVector{Sample{Tv,Tw}}`)

### Constructors

- `SampleBuffer(samples)`

    Wrap an existing `StructVector{Sample}` as a `SampleBuffer`. The input
    storage is used directly without copying and must already reside on
    the desired backend.


- `SampleBuffer(values, weights)`

    Construct a `SampleBuffer` from separate value and weight vectors.
    The two vectors must have the same length and compatible backends.
    Internally, the data is stored in a structure-of-arrays representation.


- `SampleBuffer(backend, valuetype, weighttype, size)`

    Allocate a new `SampleBuffer` of the given `size` on the specified
    `backend`, with sample values of type `valuetype` and weights of type
    `weighttype`. Storage allocation is backend-aware.

### Notes

`SampleBuffer` satisfies the full `AbstractSampleBuffer` interface and
can be used interchangeably with other sample buffer implementations in
samplers and kernel-based algorithms.
"""
struct SampleBuffer{Tv, Tw, S} <: AbstractSampleBuffer
    samples::S

    function SampleBuffer(samples::S) where {Tv, Tw, S <: StructVector{Sample{Tv, Tw}}}
        return new{Tv, Tw, S}(samples)
    end

    function SampleBuffer(vals::V, ws::W) where {Tv, Tw, V <: AbstractVector{Tv}, W <: AbstractVector{Tw}}
        S = StructArray{Sample{Tv, Tw}}((vals, ws))
        return new{Tv, Tw, typeof(S)}(S)
    end

    function SampleBuffer(backend, valuetype, weighttype, size)
        struct_arr = allocate_samples(backend, valuetype, weighttype, size)
        return new{valuetype, weighttype, typeof(struct_arr)}(struct_arr)
    end
end

Adapt.@adapt_structure SampleBuffer

@inline value_type(buf::SampleBuffer{Tv}) where {Tv} = Tv
@inline weight_type(buf::SampleBuffer{Tv, Tw}) where {Tv, Tw} = Tw

KernelAbstractions.get_backend(buf::SampleBuffer) = get_backend(buf.samples.value)

Base.length(buf::SampleBuffer) = length(buf.samples)
Base.eltype(buf::SampleBuffer{Tv, Tw}) where {Tv, Tw} = Sample{Tv, Tw}
Base.getindex(buf::SampleBuffer, idx) = buf.samples[idx]
function Base.setindex!(
        buf::SampleBuffer{Tv, Tw},
        sample::Sample{Tv, Tw},
        idx,
    ) where {Tv, Tw}
    return buf.samples[idx] = sample
end

getsample(buf::SampleBuffer, idx) = buf[idx]
setsample!(buf::SampleBuffer, sample, idx) = (buf[idx] = sample)

getvalue(buf::SampleBuffer, idx) = buf.samples.value[idx]
setvalue!(buf::SampleBuffer, value, idx) = (buf.samples.value[idx] = value)

getweight(buf::SampleBuffer, idx) = buf.samples.weight[idx]
setweight!(buf::SampleBuffer, weight, idx) = (buf.samples.weight[idx] = weight)
