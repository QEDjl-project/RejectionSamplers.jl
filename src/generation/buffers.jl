struct EventBatchBuffers{T, U, A, V}
    args::A   # proposals
    probs::V  # random numbers for rejection
    vals::V   # target values

    function EventBatchBuffers(args::A, probs::V, vals::V) where {T, U, A <: AbstractVector{T}, V <: AbstractVector{U}}
        return new{T, U, A, V}(args, probs, vals)
    end
end

function _allocate_batch_buffer(eg::EventGenerator, batch_size)
    return _allocate_batch_buffer(
        get_backend(eg),
        input_type(eg),
        output_type(eg),
        batch_size
    )
end

function _allocate_batch_buffer(backend, in_type, out_type, batch_size)
    return EventBatchBuffers(
        allocate(backend, in_type, (batch_size,)),
        allocate(backend, out_type, (batch_size,)),
        allocate(backend, out_type, (batch_size,))
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

function _allocate_output_buffer(eg::EventGenerator, batch_size, res_size)
    return _allocate_output_buffer(
        get_backend(eg),
        input_type(eg),
        output_type(eg),
        batch_size,
        res_size
    )
end

function _allocate_output_buffer(backend, in_type, out_type, batch_size, res_size)
    out_size = res_size + batch_size
    return EventOutputBuffers(
        allocate(backend, in_type, (out_size,)),
        allocate(backend, out_type, (out_size,)),
        KernelAbstractions.zeros(backend, UInt32, 1)
    )
end
