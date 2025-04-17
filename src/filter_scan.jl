"""
    filter_scan

KernelAbstractions kernel. Filters the `payload` array by comparing the respective
elements of `weights < randoms`.
Fills the indices `out_size - out_size + N` of the given buffers `out_payload` and `out_weights` with all accepted values,
where N is the number of accepted values. Also assigns `out_size += N`.

## Parameters
- `payload`: array of the payload
- `weights`: array of weights (normalized to [0,1], "probabilities")
- `randoms`: array of uniform random scalar values [0,1]
- `out_payload`: buffer for accepted payload to be written
- `out_weights`: buffer for accepted weights to be written
- `out_size`: a global memory variable containing the current number of elements in `out_payload` and `out_weights`

!!! warn
    No inbounds checks are performed anywhere! Take care that `payload`, `weights` and `randoms` are
    all of the same length `M`, and that `out_payload` and `out_weights` have at least `M+out_size` length.

!!! warn
    `out_size` must be set correctly (to the number of set elements in `out_payload` and `out_weights`) before
    calling the kernel.
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
    global_accepted_count = @localmem Int32 (1,)

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

    if weight < random
        local_accepted_idx[1] = Atomix.@atomic local_accepted_count[1] += 1
    end
    @synchronize

    # increase global output buffer index
    if thread_idx == 1
        temp = local_accepted_count[1]
        global_accepted_count[1] = Atomix.@atomic out_size[1] += temp
        #local_accepted_idx[1] = Atomix.@atomic out_size[1] += temp
        # this seems pointless but there doesn't seem to be an atomicadd that returns
        # the previous value in Atomix currently
        global_accepted_count[1] -= local_accepted_count[1]
    end
    @synchronize

    # flush to global output
    if local_accepted_idx[1] != -one(Int32)
        idx1 = global_accepted_count[1] + local_accepted_idx[1]
        out_payload[idx1] = payload[global_idx]
        out_weights[idx1] = weights[global_idx]
    end
end
