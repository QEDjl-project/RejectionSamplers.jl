function generate_event_batch!(
        eg::EventGenerator,
        batch::EventBatchBuffers,
        output::EventOutputBuffers,
    )

    backend = get_backend(eg)

    @inline generate_proposals!(eg, batch)
    KernelAbstractions.synchronize(backend)

    @inline generate_probabilities!(eg, batch)
    KernelAbstractions.synchronize(backend)

    @inline evaluate_target!(eg, batch)
    KernelAbstractions.synchronize(backend)

    rejection_filter!(eg, batch, output)
    return nothing
end

function generate_events(
        eg::EventGenerator,
        res_size,
        batch_size,
    )
    # Allocate batch buffers
    batch = _allocate_batch_buffer(eg, batch_size)

    # Allocate output buffers
    output = _allocate_output_buffer(eg, batch_size, res_size)

    # Main loop
    while true
        generate_event_batch!(eg, batch, output)

        if Vector(output.current_size)[1] >= res_size
            break
        end
    end

    # Copy back results
    out_args = Vector(output.args[1:res_size])
    out_vals = Vector(output.vals[1:res_size])

    return out_args, out_vals
end
