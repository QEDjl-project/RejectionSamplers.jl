
# generic fallbacks

# use the default rng for given dest-array and fall back to the interface
function Random.rand!(proposal::AbstractProposal{T}, dest::AbstractVector{T}) where {T}
    rng = default_rng(dest)
    _rand!(rng, proposal, dest)
end

# transformation based proposal

function _rand!(
    rng::AbstractRNG,
    proposal::AbstractTransformationBasedProposal{T},
    dest::AbstractVector{T},
) where {T}
    _rand!(rng, UnitCoordSampler(proposal), dest)
    transform!(proposal, dest)

    return dest
end
