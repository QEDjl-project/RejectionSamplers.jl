function generate_event_batch!(
        target_dist,
        proposal_dist,
        max_val,
        batch::EventBatchBuffers,
        output::EventOutputBuffers,
    )

    BACKEND = get_backend(batch.args)

    @inline generate_proposals!(proposal_dist, batch)
    KernelAbstractions.synchronize(BACKEND)

    @inline generate_probabilities!(batch)
    KernelAbstractions.synchronize(BACKEND)

    @inline evaluate_target!(target_dist, batch)
    KernelAbstractions.synchronize(BACKEND)

    rejection_filter!(batch, output, max_val)
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
    # Allocate batch buffers
    batch = _allocate_batch_buffer(backend, dtype, out_dtype, batch_size)

    # Allocate output buffers
    output = _allocate_output_buffer(backend, dtype, out_dtype, batch_size, res_size)

    # Main loop
    while true
        generate_event_batch!(dist, proposal, max_val, batch, output)

        if Vector(output.current_size)[1] >= res_size
            break
        end
    end

    # Copy back results
    out_args = Vector(output.args[1:res_size])
    out_vals = Vector(output.vals[1:res_size])

    return out_args, out_vals
end
