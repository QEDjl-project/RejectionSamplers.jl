## interface
#
abstract type AbstractSampler{Ts, Tw} end

# TODO: is this really necessary?
function degree_of_freedom end

"""
    _rand_from_host!(
        host_side_rng,
        sampler,
        backend,
        buf
    )
"""
function _rand_from_host! end


"""
    _rand_from_device!(
        device_side_rng,
        sampler,
        backend,
        buf
    )

Optional
"""
function _rand_from_device! end

"""
    _rand_single(
        device_side_rng,
        sampler
)

Optional
"""
function _rand_single end

"""
    allocate_buffer(
        rng,
        backend,
        sampler::AbstractSampler{Tv,Tw},
        size
)


Optional
"""
function allocate_buffer end
