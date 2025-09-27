function generate_proposals!(eg::EventGenerator, batch::EventBatchBuffers)
    proposal = proposal_distribution(eg)
    # TODO: update by using eg directly
    rand!(proposal, batch.args)
    return nothing
end

function generate_probabilities!(eg::EventGenerator, batch::EventBatchBuffers)
    # TODO: update by using eg directly
    rand!(batch.probs)
    return nothing
end

function evaluate_target!(eg::EventGenerator, batch::EventBatchBuffers)
    dist = target_distribution(eg)

    # TODO: update by using eg directly
    _compute!(dist, batch.vals, batch.args)
    return nothing
end

function rejection_filter!(
        eg::EventGenerator,
        batch::EventBatchBuffers,
        output::EventOutputBuffers,
    )

    batch.vals ./= maximum_value(eg)

    backend = get_backend(eg)
    filter_scan(backend, 32)(
        batch.args,
        batch.vals,
        batch.probs,
        output.args,
        output.vals,
        output.current_size;
        ndrange = length(batch.args),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
