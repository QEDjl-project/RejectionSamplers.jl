## interface
#
abstract type AbstractSampler{Ts, Tw} end

function degree_of_freedom end

"""
    _rand_from_host!(
        host_side_rng,
        sampler,
        backend,
        dest,
        u_buffer = nothing
    )
"""
function _rand_from_host! end


"""
    _rand_from_device!(
        device_side_rng,
        sampler,
        backend,
        dest
    )
"""
function _rand_from_device! end

"""
    _rand_single(
        device_side_rng,
        sampler
)

"""
function _rand_single end

### RNG strategies

@traitdef IsHostRNG{X}
@traitimpl IsHostRNG{GPUArrays.RNG}


@traitfn function _rand_backend!(
        rng::R,
        sampler::AbstractSampler{Ts, Tw},
        samples::AbstractVector,
        backend::KernelAbstractions.Backend;
        u_buf = nothing
    ) where {Ts, Tw, R <: AbstractRNG; IsHostRNG{R}}

    _rand_from_host!(rng, sampler, backend, samples, u_buf)

    return nothing
end

@traitfn function _rand_backend!(
        rng::R,
        sampler::AbstractSampler,
        samples::AbstractVector,
        backend::KernelAbstractions.Backend,
        u_buf = nothing
    ) where {{R <: AbstractRNG; !IsHostRNG{R}}}

    _rand_from_device!(rng, sampler, backend, samples)

    return nothing
end


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
        backend::KernelAbstractions.Backend,
        u_buf = nothing
    )
    _rand_backend(rng, sampler, samples, backend, u_buf)
    return nothing
end

# CPU version: needs _rand_single to be implemented
function _rand_backend!(
        rng::Random.AbstractRNG,
        sampler::AbstractSampler,
        samples::AbstractVector,
        ::KernelAbstractions.CPU,
        u_buf = nothing
    )
    @inbounds for i in eachindex(samples)
        single_sample = _rand_single(rng, sampler)
        samples.value[i], samples.weight[i] = single_sample.value, single_sample.weight
    end
    return nothing
end

# NOTE:
# -
@kernel function _rand_gpu_kernel(rng, samples, sampler::AbstractSampler)
    idx = @index(Global, Linear)
    sample = _rand_single(rng, sampler)
    @inbounds samples.value[idx] = sample.value
    @inbounds samples.weight[idx] = sample.weight
end

# generic implementation which falls back to _rand_single
function _rand_from_device!(
        rng::AbstactRNG,
        sampler::AbstractSampler,
        backend::KernelAbstractions.Backend,
        samples::AbstractVector,
    )
    _rand_gpu_kernel(backend, 32)(
        rng, samples, sampler;
        ndrange = size(samples)
    )
    return nothing
end
