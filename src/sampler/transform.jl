abstract type AbstractTransformBasedSampler{Tv, Tw} <: AbstractSampler{Tv, Tw} end


"""
    _transform_value(sampler::AbstractTransformBasedSampler{Tv, Tw}, unit_input)

Interface function: transform unit random numbers into the target sample space.
Return the transformed value.
"""
function _transform_value end

"""
    _transform_weight(sampler::AbstractTransformBasedSampler{Tv, Tw}, unit_input, value)

Interface function: return weight for the transformation [`_transform_value`](@ref) for given `unit_input` and `value` in sample space.
"""
function _transform_weight end

"""
    _transform!(sampler::AbstractTransformBasedSampler{Tv, Tw}, buffer, unit_rng_nums)
        -> nothing

Batch version of [`transform`](@ref).
"""
function _transform!(sampler::AbstractTransformBasedSampler, buf, vs)
    buf.samples.value .= _transform_value.(sampler, vs)
    buf.samples.weight .= _transform_weight.(sampler, vs, buf.samples.value)

    return nothing
end


# device-side RNG can transform every sample directly
function _rand_single(rng::AbstractRNG, sampler::AbstractTransformBasedSampler{Tv}) where {Tv}
    u = rand(rng, Tv)
    value = _transform_value(sampler, u)
    weight = _transform_weight(sampler, u, value)
    return Sample(value, weight)
end


function _rand_from_host!(
        rng::AbstractRNG,
        sampler::AbstractTransformBasedSampler{Tv, Tw},
        backend::KernelAbstractions.Backend,
        buf::AbstractSampleBuffer
    ) where {Tv, Tw}
    N = degrees_of_freedom(sampler)

    rnd_nums = allocate(backend, Tv, length(buf))
    rand!(rng, rnd_nums)

    _transform!(sampler, buf, rnd_nums)

    return nothing
end
