# TODO:
# - use the mock proposal and implement the new interface

abstract type AbstractMockSampler{TT, T} <: RejectionSamplers.AbstractSampler{TT, T} end
abstract type AbstractMockTransformbasedSampler{TT, T} <: RejectionSamplers.AbstractTransformBasedSampler{TT, T} end

function RejectionSamplers.allocate_buffer(
        rng::AbstractRNG,
        sampler::Union{AbstractMockSampler{TT, T}, AbstractMockTransformbasedSampler{TT, T}},
        backend::KernelAbstractions.Backend,
        size
    ) where {T, TT}
    return SampleBuffer(backend, TT, T, size)
end

struct MockSampler{T, D, TT} <: AbstractMockSampler{TT, T}
    weight::T

    # scalar sample
    MockSampler(::Type{T}, w::T) where {T <: Real} = new{T, 1, T}(w)

    # SVector sample
    MockSampler(::Type{TT}, w::T) where {N, T <: Real, TT <: SVector{N, T}} = new{T, N, TT}(w)

    # NTuple sample
    MockSampler(::Type{TT}, w::T) where {N, T <: Real, TT <: NTuple{N, T}} = new{T, N, TT}(w)
end

RejectionSamplers.degrees_of_freedom(::MockSampler{T, D, TT}) where {T, D, TT} = D

### Sample functions
# - these functions should work in both cases, host-side and device-side
# - the sampling is the same for every RNG passed
# - the samples are deterministically determined by the length of the buffer and a
# constant weight stored in the sampler

# Scalar
function _mock_rand!(rng, sampler::MockSampler{T}, backend, buf) where {T <: Real}
    n = length(buf)
    steps = T(2 / (n + 1))
    buf.samples.value .= StepRangeLen(-1 + steps, steps, n)
    buf.samples.weight .= KernelAbstractions.ones(backend, T, n) * sampler.weight

    return nothing
end

# SVector
function _mock_rand!(rng, sampler::MockSampler{T, D, TT}, backend, buf) where {T <: Real, D, TT <: SVector{D, T}}
    n = length(buf)
    steps = T(2 / (n + 1))
    map!(i -> fill(-1 + i * steps, TT), buf.samples.value, 1:n)
    buf.samples.weight .= KernelAbstractions.ones(backend, T, n) * sampler.weight
    return nothing
end

# NTuple
# WARN: this only works on GPU for D<=10
# (because of unrolled tuple construction in `ntuple()`)
function _mock_rand!(rng, sampler::MockSampler{T, D, TT}, backend, buf) where {T <: Real, D, TT <: NTuple{D, T}}
    n = length(buf)
    steps = T(2 / (n + 1))
    map!(i -> ntuple(x -> -1 + i * steps, Val(D)), buf.samples.value, 1:n)
    buf.samples.weight .= KernelAbstractions.ones(backend, T, n) * sampler.weight
    return nothing
end

function RejectionSamplers._rand_from_device!(
        rng::AbstractRNG,
        sampler::MockSampler,
        backend::KernelAbstractions.Backend,
        buf::AbstractSampleBuffer
    )
    return _mock_rand!(rng, sampler, backend, buf)
end

function RejectionSamplers._rand_from_host!(
        rng::AbstractRNG,
        sampler::MockSampler,
        backend::KernelAbstractions.Backend,
        buf::AbstractSampleBuffer
    )
    return _mock_rand!(rng, sampler, backend, buf)
end

# implements only `_rand_single`
struct MockSamplerSingle{T, N, TT} <: AbstractMockSampler{TT, T}
    weight::T

    # scalar sample
    MockSamplerSingle(::Type{T}, w::T) where {T <: Real} = new{T, 1, T}(w)

    # SVector sample
    MockSamplerSingle(::Type{TT}, w::T) where {N, T <: Real, TT <: SVector{N, T}} = new{T, N, TT}(w)

    # NTuple sample
    MockSamplerSingle(::Type{TT}, w::T) where {N, T <: Real, TT <: NTuple{N, T}} = new{T, N, TT}(w)
end

### sampling functions
# - returns always the sample `Sample(-1, sampler.weight)`

function RejectionSamplers._rand_single(rng::AbstractRNG, sampler::MockSamplerSingle{T, 1}) where {T <: Real}
    return Sample(-one(T), sampler.weight)
end

function RejectionSamplers._rand_single(rng::AbstractRNG, sampler::MockSamplerSingle{T, N, TT}) where {N, T <: Real, TT <: SVector{N, T}}
    s = @SVector fill(-one(T), N)

    return Sample(
        s,
        sampler.weight
    )
end

function RejectionSamplers._rand_single(rng::AbstractRNG, sampler::MockSamplerSingle{T, N, TT}) where {N, T <: Real, TT <: NTuple{N, T}}
    return Sample(
        ntuple(x -> -one(T), Val(N)),
        sampler.weight
    )
end

### transform based sampler
struct MockTransformBasedSampler{T, D, TT} <: AbstractMockTransformbasedSampler{TT, T}
    weight::T

    # scalar sample
    MockTransformBasedSampler(::Type{T}, w::T) where {T <: Real} = new{T, 1, T}(w)

    # SVector sample
    MockTransformBasedSampler(::Type{TT}, w::T) where {N, T <: Real, TT <: SVector{N, T}} = new{T, N, TT}(w)

    # NTuple sample
    MockTransformBasedSampler(::Type{TT}, w::T) where {N, T <: Real, TT <: NTuple{N, T}} = new{T, N, TT}(w)
end

RejectionSamplers.degrees_of_freedom(::MockTransformBasedSampler{T, D, TT}) where {T, D, TT} = D

@inline function RejectionSamplers._transform_value(
        sampler::MockTransformBasedSampler{T, 1},
        v::T,
    ) where {T}
    w = sampler.weight
    return w
end

@inline function RejectionSamplers._transform_value(
        sampler::MockTransformBasedSampler{T, N},
        v::SVector{N, T},
    ) where {N, T}
    w = sampler.weight
    return SVector{N, T}(
        ntuple(
            x -> w,
            Val(N),
        ),
    )
end


@inline function RejectionSamplers._transform_value(
        sampler::MockTransformBasedSampler{T, N},
        v::NTuple{N, T},
    ) where {N, T}
    w = sampler.weight
    return ntuple(x -> w, Val(N))
end

@inline function RejectionSamplers._transform_weight(
        sampler::MockTransformBasedSampler{T},
        v,
        value::T
    ) where {T}
    return one(T)
end

@inline function RejectionSamplers._transform_weight(
        sampler::MockTransformBasedSampler{T, N},
        v,
        value::NTuple{N, T}
    ) where {T, N}
    return one(T)
end

@inline function RejectionSamplers._transform_weight(
        sampler::MockTransformBasedSampler{T, N},
        v,
        value::SVector{N, T}
    ) where {T, N}
    return one(T)
end
