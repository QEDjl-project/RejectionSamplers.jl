# TODO:
# - use the mock proposal and implement the new interface

struct MockSampler{T, D, TT} <: RejectionSamplers.AbstractSampler{TT, T}
    weight::T

    # scalar sample
    MockSampler(::Type{T}, w::T) where {T <: Real} = new{T, 1, T}(w)

    # SVector sample
    MockSampler(::Type{TT}, w::T) where {N, T <: Real, TT <: SVector{N, T}} = new{T, N, TT}(w)

    # NTuple sample
    MockSampler(::Type{TT}, w::T) where {N, T <: Real, TT <: NTuple{N, T}} = new{T, N, TT}(w)
end

RejectionSamplers.degrees_of_freedom(::MockSampler{T, D, TT}) where {T, D, TT} = D

### overwriting the generic because MockSampler is not random


# scalar
function RejectionSamplers._rand!(
        rng::AbstractRNG,
        sampler::MockSampler{T, 1},
        samples::AbstractVector,
        backend::KernelAbstractions.Backend
    ) where {T <: Real}

    n = length(sample_dest)
    steps = T(2 / (n + 1))
    samples.value .= StepRangeLen(-1 + steps, steps, n)
    samples.weight .= KernelAbstractions.ones(backend, T, n) * sampler.weight

    return nothing
end

# SVector
function RejectionSamplers._rand!(
        rng::AbstractRNG,
        sampler::MockSampler{T, D, TT},
        samples::AbstractVector,
        backend::KernelAbstractions.Backend
    ) where {T <: Real, D, TT <: SVector{D, T}}

    n = length(samples)
    steps = T(2 / (n + 1))
    map!(i -> fill(-1 + i * steps, TT), samples.value, 1:n)
    samples.weight .= KernelAbstractions.ones(backend, T, n) * sampler.weight
    return nothing
end

# NTuple
# WARN: this only works on GPU for D<=10
# (because of unrolled tuple construction in `ntuple()`)
function RejectionSamplers._rand!(
        rng::AbstractRNG,
        sampler::MockSampler{T, D, TT},
        samples::AbstractVector,
        backend::KernelAbstractions.Backend
    ) where {T <: Real, D, TT <: NTuple{D, T}}

    n = length(sample_dest)
    steps = T(2 / (n + 1))
    map!(i -> ntuple(x -> -1 + i * steps, D), samples.value, 1:n)
    samples.weight .= KernelAbstractions.ones(backend, T, n) * sampler.weight
    return nothing
end

# implements only `_rand_single`
struct MockSamplerSingle{T, N, TT} <: RejectionSamplers.AbstractSampler{TT, T}
    weight::T

    # scalar sample
    MockSamplerSingle(::Type{T}, w::T) where {T <: Real} = new{T, 1, T}(w)

    # SVector sample
    MockSamplerSingle(::Type{TT}, w::T) where {N, T <: Real, TT <: SVector{N, T}} = new{T, N, TT}(w)

    # NTuple sample
    MockSamplerSingle(::Type{TT}, w::T) where {N, T <: Real, TT <: NTuple{N, T}} = new{T, N, TT}(w)
end

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
        ntuple(x -> -one(T), N),
        sampler.weight
    )
end
