# TODO: add this to `filter_scan`
function filter_scan_update!(
        d_args,
        d_vals,
        d_probs,
        d_out_args,
        d_out_vals,
        current_out_size,

        # TODO: add this one
        max_val,
    )

    # TODO: prettify!
    @inline d_vals ./= max_val

    BACKEND = get_backend(d_args)
    filter_scan(BACKEND, 32)(
        d_args,
        d_vals,
        d_probs,
        d_out_args,
        d_out_vals,
        current_out_size;
        ndrange = length(d_args),
    )

    return KernelAbstractions.synchronize(BACKEND)
end

# generate event batches
function _generate_events!(
        target_dist,
        proposal_dist,
        max_val,
        d_args,
        d_probs,
        d_vals,
        d_out_args,
        d_out_vals,
        current_out_size,
    )
    BACKEND = get_backend(d_args)

    # TODO:
    # - write this as a wrapper around a KA Kernel
    @inline rand!(proposal_dist, d_args)
    @inline KernelAbstractions.synchronize(BACKEND)

    # TODO:
    # - write this as a wrapper around a KA Kernel
    @inline rand!(d_probs)
    @inline KernelAbstractions.synchronize(BACKEND)

    # TODO:
    # - write this as a wrapper around a KA Kernel
    @inline _compute!(target_dist, d_vals, d_args)
    @inline KernelAbstractions.synchronize(BACKEND)

    @inline filter_scan_update!(
        d_args,
        d_vals,
        d_probs,
        d_out_args,
        d_out_vals,
        current_out_size, # rename d_...
        max_val,
    )

    return nothing
end


function generate_events(
        dist,
        proposal,
        max_val,
        res_size,
        batch_size,
        backend,
        out_dtype,
        dtype = out_dtype,
    )

    # allocate input buffer on GPU (batch_size)
    d_args = allocate(backend, dtype, (batch_size,))
    d_probs = allocate(backend, out_dtype, (batch_size,))
    d_vals = allocate(backend, out_dtype, (batch_size,))

    # allocate output buffer on GPU (out_size = res_size + batch_size)
    out_size = res_size + batch_size
    d_out_args = allocate(backend, dtype, (out_size,))
    d_out_vals = allocate(backend, out_dtype, (out_size,))

    # allocate current_out_size + init with zero
    current_out_size = KernelAbstractions.zeros(backend, UInt32, 1)

    #d_max_val = allocate(backend, dtype, (1,))
    #copyto!(d_max_val, [max_val])

    running = true
    while running
        @inline _generate_events!(
            dist,
            proposal,
            max_val,
            d_args,
            d_probs,
            d_vals,
            d_out_args,
            d_out_vals,
            current_out_size,
        )

        h_current_out_size = Vector(current_out_size)[1]
        if h_current_out_size >= res_size
            running = false
        end
    end

    # check if slicing works on GPU
    # if not, use copyto!(dest,dest_idx,d_arr,d_idx)
    out_args = Vector(d_out_args[1:res_size])
    out_vals = Vector(d_out_vals[1:res_size])

    return out_args, out_vals

end
