@inline value_type(::AbstractSample{Ts}) where {Ts} = Ts
@inline weight_type(::AbstractSample{Ts, Tw}) where {Ts, Tw} = Tw
