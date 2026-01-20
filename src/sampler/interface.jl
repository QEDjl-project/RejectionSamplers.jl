## interface
#
abstract type AbstractSampler{Ts, Tw} end

function degree_of_freedom end

"""
    _rand_single(rng::AbstactRNG, sampler::AbstractSampler)::Sample

"""
function _rand_single end


## generics
#
# TODO:
# - maybe this can be unified to one _rand_on_device and one kernel for CPU and GPU
# - check if different overloads for GPU and CPU are possible
#
function _rand!(
        rng::Random.AbstractRNG,
        sampler::AbstractSampler,
        samples::AbstractVector,
        backend::KernelAbstractions.Backend
    )
    _rand_on_device!(rng, sampler, samples, backend)
    return nothing
end

# CPU version:
function _rand_on_device!(
        rng::Random.AbstractRNG,
        sampler::AbstractSampler,
        samples::AbstractVector,
        ::KernelAbstractions.CPU,
    )
    @inbounds for i in eachindex(samples)
        single_sample = _rand_single(rng, sampler)
        samples.value[i], samples.weight[i] = single_sample.value, single_sample.weight
    end
    return nothing
end

# NOTE:
# - this might only work for device rng, but not for counter based rng, because the latter
# need a thread-aware state/counter, e.g. the thread index.
# -
@kernel function _rand_gpu_kernel(rng, samples, sampler::AbstractSampler)
    idx = @index(Global, Linear)
    sample = _rand_single(rng, sampler)
    @inbounds samples.value[idx] = sample.value
    @inbounds samples.weight[idx] = sample.weight
end

@kernel function _rand_gpu_kernel(samples, sampler::AbstractSampler)
    idx = @index(Global, Linear)
    sample = _rand_single(Random.default_rng(), sampler)
    @inbounds samples.value[idx] = sample.value
    @inbounds samples.weight[idx] = sample.weight
end

function _rand_on_device!(
        sampler::AbstractSampler,
        samples::AbstractVector,
        backend::KernelAbstractions.GPU,
    )
    _rand_gpu_kernel(backend, 32)(
        samples, sampler;
        ndrange = size(samples)
    )
    return nothing
end

function _rand_on_device!(
        rng::AbstactRNG,
        sampler::AbstractSampler,
        samples::AbstractVector,
        backend::KernelAbstractions.GPU,
    )
    _rand_gpu_kernel(backend, 32)(
        rng, samples, sampler;
        ndrange = size(samples)
    )
    return nothing
end
