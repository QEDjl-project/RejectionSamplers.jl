tuple_isapprox(::NTuple{N}, ::NTuple{M}) where {N, M} = false

function tuple_isapprox(t1::NTuple{N}, t2::NTuple{N}) where {N}
    for i in eachindex(t1)
        isapprox(t1[i], t2[i]) || return false
    end
    return true
end

# version of isapprox, which supports NTuple
_isapprox(args...; kwargs...) = isapprox(args...; kwargs...)
_isapprox(t1::NTuple{N}, t2::NTuple{N}) where {N} = tuple_isapprox(t1, t2)
_isapprox(a1::AbstractVector{TT}, a2::AbstractVector{TT}) where {N, T, TT <: NTuple{N, T}} = all(tuple_isapprox.(a1, a2))

if VERSION >= v"1.11"
    _get_value_types(el_type, ::Val{N} = Val(4)) where {N} = (el_type, SVector{N, el_type}, NTuple{N, el_type})
else
    # Julia 1.10 does not support random generation of NTuple on CPU
    @warn "Julia 1.10 does not support random generation of NTuple on CPU"
    _get_value_types(el_type, ::Val{N} = Val(4)) where {N} = (el_type, SVector{N, el_type})
end
