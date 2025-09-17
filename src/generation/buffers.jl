struct EventBatchBuffers{T, U}
    args::AbstractVector{T}   # proposals
    probs::AbstractVector{U}  # random numbers for rejection
    vals::AbstractVector{U}   # target values
end

function _allocate_batch_buffer(backend, dtype, out_dtype, batch_size)
    return EventBatchBuffers(
        allocate(backend, dtype, (batch_size,)),
        allocate(backend, out_dtype, (batch_size,)),
        allocate(backend, out_dtype, (batch_size,))
    )
end

struct EventOutputBuffers{T, U}
    args::AbstractVector{T}   # accepted events
    vals::AbstractVector{U}   # accepted weights
    current_size::AbstractVector{UInt32} # size counter on device
end

function _allocate_output_buffer(backend, dtype, out_dtype, batch_size, res_size)
    out_size = res_size + batch_size
    return EventOutputBuffers(
        allocate(backend, dtype, (out_size,)),
        allocate(backend, out_dtype, (out_size,)),
        KernelAbstractions.zeros(backend, UInt32, 1)
    )
end
