struct EventBatchBuffers{T, U, A, V}
    args::A   # proposals
    probs::V  # random numbers for rejection
    vals::V   # target values

    function EventBatchBuffers(args::A, probs::V, vals::V) where {T, U, A <: AbstractVector{T}, V <: AbstractVector{U}}
        return new{T, U, A, V}(args, probs, vals)
    end
end

function _allocate_batch_buffer(backend, dtype, out_dtype, batch_size)
    return EventBatchBuffers(
        allocate(backend, dtype, (batch_size,)),
        allocate(backend, out_dtype, (batch_size,)),
        allocate(backend, out_dtype, (batch_size,))
    )
end

struct EventOutputBuffers{T, U, A, V, S}
    args::A             # accepted events
    vals::V             # accepted weights
    current_size::S     # size counter on device

    function EventOutputBuffers(args::A, vals::V, current_size::S) where {
            T, U, A <: AbstractVector{T}, V <: AbstractVector{U}, S <: AbstractVector{UInt32},
        }
        return new{T, U, A, V, S}(args, vals, current_size)
    end
end

function _allocate_output_buffer(backend, dtype, out_dtype, batch_size, res_size)
    out_size = res_size + batch_size
    return EventOutputBuffers(
        allocate(backend, dtype, (out_size,)),
        allocate(backend, out_dtype, (out_size,)),
        KernelAbstractions.zeros(backend, UInt32, 1)
    )
end
