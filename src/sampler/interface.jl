abstract type AbstractSampler{Ts, Tw} end

function degree_of_freedom end

"""
    _rand_single(rng::AbstactRNG, sampler::AbstractSampler)::Sample


for CPU
"""
function _rand_single end

"""
    _rand_on_device!(
        rng::GPUArrays.RNG,
        sampler::AbstractSampler,
        samples::SampleVector,
        backend::KernelAbstractions.Backend,
    )

for GPU
"""
function _rand_on_device! end

# TODO:
# - consider extenting the interface to allow generic implementations of
# `allocate_samples(::AbstractSampler, length)`
