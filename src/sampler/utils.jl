@inline value_type(::AbstractSampler{Ts}) where {Ts} = Ts
@inline weight_type(::AbstractSampler{Ts, Tw}) where {Ts, Tw} = Tw

# TODO: update this by directly allocating a buffer
# convenience function for allocating samples
@inline function allocate_samples(backend, ::AbstractSampler{Ts, Tw}, batch_size) where {Ts, Tw}
    return allocate_samples(backend, Ts, Tw, batch_size)
end
