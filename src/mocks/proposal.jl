struct MockProposal{T, D, TT, V} <: RejectionSamplers.AbstractProposalDistribution{TT, T, V}

    # scalar proposal
    MockProposal(::Type{T}) where {T <: Real} = new{T, 1, T, Distributions.Univariate}()

    # SVector proposal
    MockProposal(::Type{TT}) where {N, T <: Real, TT <: SVector{N, T}} = new{T, N, TT, RejectionSamplers.CoordinateVariate}()

    # NTuple proposal
    MockProposal(::Type{TT}) where {N, T <: Real, TT <: NTuple{N, T}} = new{T, N, TT, RejectionSamplers.CoordinateVariate}()
end

RejectionSamplers.degrees_of_freedom(::MockProposal{T, D, TT}) where {T, D, TT} = D

### overwriting the generic because MockProposal is not random
# TODO: adjust proposal interface, maybe use `generate_proposal!` as the interface
# function and call `Random.rand!` or `_rand!` per default.

# scalar
function RejectionSamplers._propose!(
        rng::AbstractRNG,
        ::MockProposal{T, 1},
        sample_dest::AbstractVector{T},
        weight_dest::AbstractVector{T},
        backend::KernelAbstractions.Backend
    ) where {T <: Real}

    n = length(sample_dest)
    steps = T(2 / (n + 1))
    sample_dest .= StepRangeLen(-1 + steps, steps, n)
    weight_dest .= KernelAbstractions.ones(backend, T, n)
    return nothing
end

# SVector
function RejectionSamplers._propose!(
        rng::AbstractRNG,
        ::MockProposal{T, D, TT, RejectionSamplers.CoordinateVariate},
        sample_dest::AbstractVector{TT},
        weight_dest::AbstractVector{T},
        backend::KernelAbstractions.Backend
    ) where {T <: Real, D, TT <: SVector{D, T}}

    n = length(sample_dest)
    steps = T(2 / (n + 1))
    map!(i -> fill(-1 + i * steps, TT), sample_dest, 1:n)
    weight_dest .= KernelAbstractions.ones(backend, T, n)
    return nothing
end

# NTuple
# WARN: this only works on GPU for D<=10
# (because of unrolled tuple construction in `ntuple()`)
function RejectionSamplers._propose!(
        rng::AbstractRNG,
        ::MockProposal{T, D, TT, RejectionSamplers.CoordinateVariate},
        sample_dest::AbstractVector{TT},
        weight_dest::AbstractVector{T},
        backend::KernelAbstractions.Backend
    ) where {T <: Real, D, TT <: NTuple{D, T}}

    n = length(sample_dest)
    steps = T(2 / (n + 1))
    map!(i -> ntuple(x -> -1 + i * steps, D), sample_dest, 1:n)
    return weight_dest .= KernelAbstractions.ones(backend, T, n)
end
