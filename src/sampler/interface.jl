## interface
#

"""
    AbstractSampler{Tv,Tw}

Abstract base type for the sampler interface. These abstract samplers that generate random
samples on CPU and/or accelerator backends.

An `AbstractSampler` represents a sampling rule or distribution, independent
of the execution location. Concrete subtypes must implement at least one
sampling path via the methods described below.

# Type Parameters
- `Tv`: element type of the generated values.
- `Tw`: element type of the weight associated with the generated value.

# Required Interface
Every concrete sampler must implement:

- `_rand_from_host!(host_side_rng, sampler, backend, buf)`

This method generates samples using a host-side RNG and writes them into
`buf`. It must work for all supported backends, including those where the
destination buffer resides on an accelerator device.

# Optional Interface
Samplers may additionally implement one or more of the following methods
to enable more efficient or more specialized sampling strategies:

- `_rand_from_device!(device_side_rng, sampler, backend, buf)`

  Generates samples using a device-side RNG, typically by launching kernels
  on the backend. This allows fully device-resident sampling.

- `_rand_single(device_side_rng, sampler)`

  Generates a single sample, intended for use inside device kernels.

- `allocate_buffer(rng, backend, sampler, size)`

  Allocates a sampler-specific `AbstractSampleBuffer` for storing samples.
  This can be used to control memory layout, placement, or auxiliary storage.

"""
abstract type AbstractSampler{Ts, Tw} end

"""
    _rand_from_host!(host_side_rng, sampler, backend, buf)

Interface function: fill `buf` with samples generated using a host-side RNG.

This method is mandatory for every sampler. It is intended for RNGs that
can only be invoked on the host but are capable of writing directly to
device-resident memory (e.g. `rand!(rng, CuArray)`).

The implementation is responsible for generating as many samples as
required by `buf`, using `backend` to determine the execution context.
"""
function _rand_from_host! end


"""
    _rand_from_device!(device_side_rng, sampler, backend, buf)

Interface function: fill `buf` with samples generated using a device-side RNG.

This method is optional. It is intended for RNGs that can be invoked
directly from within device kernels (e.g. GPU kernels). Implementations
may launch kernels using `backend` and generate samples entirely on
the device.
"""
function _rand_from_device! end

"""
    _rand_single(device_side_rng, sampler) -> sample

Interface function: generate a single sample using a device-side RNG.

This optional method is intended for use inside device kernels, where
buffer-based sampling is not appropriate. It should return one sample
drawn according to the distribution represented by `sampler`.
"""
function _rand_single end

"""
    allocate_buffer(rng, backend, sampler::AbstractSampler{Tv,Tw}, size) -> buf

interface function: allocate and return an `AbstractSampleBuffer` suitable for storing `size`
samples produced by `sampler`.

This optional method allows samplers to control the concrete buffer type
and memory placement (host or device) based on the RNG and `backend`.
If not implemented, a default buffer allocation strategy is expected
to be used by the caller.
"""
function allocate_buffer end
