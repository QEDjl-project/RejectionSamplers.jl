"""
    filter_scan(payload, weights, randoms, out_payload, out_weights, out_size)

KernelAbstractions kernel. Filters the `payload` array by comparing the respective
elements of `weights < randoms`.
Fills the indices `out_size - out_size + N` of the given buffers `out_payload` and `out_weights` with all accepted values,
where N is the number of accepted values. Also assigns `out_size += N`.

## Parameters
- input `payload`: array of the payload
- input `weights`: array of weights (normalized to [0,1]; outliers allowed, see `out_weights`; "probabilities")
- input `randoms`: array of uniform random independent scalar values [0,1]
- output `out_payload`: buffer for accepted payload to be written
- output `out_weights`: buffer for accepted weights to be written; every accepted weight will be 1.0, or the original weight if it was larger than 1.0
- input/output `out_size`: a global memory variable (array of length 1) containing the current number of elements in `out_payload` and `out_weights`

!!! warn
    No inbounds checks are performed anywhere! Take care that `payload`, `weights` and `randoms` are
    all of the same length `M`, and that `out_payload` and `out_weights` have at least `M+out_size` length.

!!! warn
    `out_size` must be set correctly (to the number of set elements in `out_payload` and `out_weights`) before
    calling the kernel. Also note that `Int64` is not supported on some backends (Metal.jl and oneAPI.jl).
"""
@kernel inbounds = true function filter_scan(
        @Const(payload),
        @Const(weights),
        @Const(randoms),
        out_payload,
        out_weights,
        out_size,
    )
    local_accepted_count = @localmem Int32 (1,)
    global_accepted_idx = @localmem Int32 (1,)

    global_idx = @index(Global, Linear)
    thread_idx = @index(Local, Linear)

    if thread_idx == 1
        local_accepted_count[1] = 0
    end
    @synchronize

    # filter using randoms
    weight = weights[global_idx]
    random = randoms[global_idx]

    local_accepted_idx = @private Int32 (1,)
    local_accepted_idx[1] = -one(Int32)

    if random < weight
        local_accepted_idx[1] = Atomix.@atomic local_accepted_count[1] += 1
    end
    @synchronize

    # increase global output buffer index
    if thread_idx == 1
        temp = local_accepted_count[1]
        global_accepted_idx[1] = Atomix.@atomic out_size[1] += temp
        # this seems pointless but there doesn't seem to be an atomicadd that returns
        # the previous value in Atomix currently
        global_accepted_idx[1] -= local_accepted_count[1]
    end
    @synchronize

    # flush to global output
    if local_accepted_idx[1] != -one(Int32)
        idx1 = global_accepted_idx[1] + local_accepted_idx[1]
        out_payload[idx1] = payload[global_idx]

        # update weights
        out_weights[idx1] = if weights[global_idx] > one(eltype(weights))
            weights[global_idx]     # use weight itself if it was > 1
        else
            one(eltype(weights))    # use 1 otherwise
        end
    end
end
