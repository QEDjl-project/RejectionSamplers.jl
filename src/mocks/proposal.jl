struct MockProposal{T, D, TT, V} <: GPUEventGenerators.AbstractProposal{TT, V}

    # scalar proposal
    MockProposal(::Type{T}) where {T <: Real} = new{T, 1, T, Distributions.Univariate}()

    # SVector proposal
    MockProposal(::Type{TT}) where {N, T <: Real, TT <: SVector{N, T}} = new{T, N, TT, GPUEventGenerators.CoordinateVariate}()

    # NTuple proposal
    MockProposal(::Type{TT}) where {N, T <: Real, TT <: NTuple{N, T}} = new{T, N, TT, GPUEventGenerators.CoordinateVariate}()
end

GPUEventGenerators.degrees_of_freedom(::MockProposal{T, D, TT}) where {T, D, TT} = D

### overwriting the generic because MockProposal is not random
# TODO: adjust proposal interface, maybe use `generate_proposal!` as the interface
# function and call `Random.rand!` or `_rand!` per default.

# scalar
function Random.rand!(::MockProposal{T, 1}, dest::AbstractVector{T}) where {T}
    n = length(dest)
    steps = T(2 / (n + 1))
    dest .= StepRangeLen(-1 + steps, steps, n)
    return dest
end

# SVector
function Random.rand!(::MockProposal{T, D, TT}, dest::AbstractVector{TT}) where {T <: Real, D, TT <: SVector{D, T}}
    n = length(dest)
    steps = T(2 / (n + 1))
    map!(i -> fill(-1 + i * steps, TT), dest, 1:n)
    return dest
end

# NTuple
# WARN: this only works on GPU for D<=10
# (because of unrolled tuple construction in `ntuple()`)
function Random.rand!(::MockProposal{T, D, TT}, dest::AbstractVector{TT}) where {T, D, TT <: NTuple}
    n = length(dest)
    steps = T(2 / (n + 1))
    map!(i -> ntuple(x -> -1 + i * steps, D), dest, 1:n)
    return dest
end
