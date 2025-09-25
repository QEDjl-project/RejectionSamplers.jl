function generate_proposals!(proposal, batch::EventBatchBuffers)
    rand!(proposal, batch.args)
    return nothing
end

function generate_probabilities!(batch::EventBatchBuffers)
    rand!(batch.probs)
    return nothing
end

function evaluate_target!(dist, batch::EventBatchBuffers)
    _compute!(dist, batch.vals, batch.args)
    return nothing
end

function rejection_filter!(
        batch::EventBatchBuffers,
        output::EventOutputBuffers,
        max_val
    )
    batch.vals ./= max_val

    BACKEND = get_backend(batch.args)
    filter_scan(BACKEND, 32)(
        batch.args,
        batch.vals,
        batch.probs,
        output.args,
        output.vals,
        output.current_size;
        ndrange = length(batch.args),
    )
    KernelAbstractions.synchronize(BACKEND)
    return nothing
end
