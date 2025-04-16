"""
    filter_scan

KernelAbstractions kernel. Filters the `in_Q` array of payload by comparing the respective
elements `in_T < in_RNG`.
Fills the first `g_out_size` values of the given buffers `out_Q` and `out_T` with all accepted values.

## Parameters
- `in_Q`: array of the payload
- `in_T`: array of weights (normalized to [0,1], "probabilities")
- `in_RNG`: array of uniform random scalar values [0,1]
- `out_Q`: buffer for accepted payload to be written
- `out_T`: buffer for accepted weights to be written
- `g_out_size`: a global memory variable containing the current number of elements in `out_Q` and `out_T`

!!! warn
    No inbounds checks are performed anywhere! Take care that `in_Q`, `in_T` and `in_RNG` are 
    all of the same length `N`, and that `out_Q` and `out_T` have at least `N+g_out_size` length.
"""
@kernel inbounds = true function filter_scan(
    @Const(in_Q),
    @Const(in_T),
    @Const(in_RNG),
    out_Q,
    out_T,
    g_out_size,
)
    count = @localmem Int32 (1,)
    g_out_idx = @localmem Int64 (1,)

    global_idx = @index(Global, Linear)
    thread_idx = @index(Local, Linear)

    # filter using in_RNG
    in_t = in_T[global_idx]
    in_rng = in_RNG[global_idx]

    old = @private Int32 (1,)
    old[1] = -one(Int32)

    if in_t < in_rng
        old[1] = Atomix.@atomic count[1] += 1
    end
    @synchronize

    # increase global output buffer size
    if thread_idx == 1
        g_out_idx[1] = Atomix.@atomic g_out_size[1] += count[1]
        # this seems pointless but there doesn't seem to be an atomicadd that returns
        # the previous value in Atomix currently
        g_out_idx[1] -= count[1]
    end
    @synchronize

    # flush to global output
    if old[1] != -one(Int32)
        idx1 = g_out_idx[1] + old[1]
        out_Q[idx1] = in_Q[global_idx]
        out_T[idx1] = in_T[global_idx]
    end
end
