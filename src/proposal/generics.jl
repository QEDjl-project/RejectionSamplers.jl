# generic fallbacks

function propose_single(proposal::AbstractProposalDistribution)
    rng = default_rng()
    return _propose_single(rng, proposal)
end

# default backend infered from array type
function propose!(
        rng::AbstractRNG,
        proposal::AbstractProposalDistribution{Ts, Tw},
        sample_dest::AbstractVector{Ts},
        weight_dest::AbstractVector{Tw},
        backend::KernelAbstractions.Backend
    ) where {Ts, Tw}

    length(sample_dest) == length(weight_dest) || throw(
        ArgumentError(
            "length of sample and weight destination arrays must be the same"
        )
    )

    return _propose!(rng, proposal, sample_dest, weight_dest, backend)
end

@inline function propose!(
        rng::AbstractRNG,
        proposal::AbstractProposalDistribution{Ts, Tw},
        sample_dest::AbstractVector{Ts},
        weight_dest::AbstractVector{Tw},
    ) where {Ts, Tw}
    backend = get_backend(sample_dest)
    return propose!(rng, proposal, sample_dest, weight_dest, backend)
end

@inline function propose!(
        proposal::AbstractProposalDistribution{Ts, Tw},
        sample_dest::SV,
        weight_dest::AbstractVector{Tw},
        backend::KernelAbstractions.Backend
    ) where {Ts, Tw, SV <: AbstractVector{Ts}}
    rng = default_rng(SV)
    return propose!(rng, proposal, sample_dest, weight_dest, backend)
end

function propose!(
        proposal::AbstractProposalDistribution{Ts, Tw},
        sample_dest::SV,
        weight_dest::AbstractVector{Tw},
    ) where {Ts, Tw, SV <: AbstractVector{Ts}}
    rng = default_rng(SV)
    backend = get_backend(sample_dest)

    return propose!(rng, proposal, sample_dest, weight_dest, backend)
end
