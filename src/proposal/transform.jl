
# transformation based proposal
abstract type AbstractTransformationBasedProposal{T,F} <: AbstractProposal{T,F} end

"""

    _transform!(proposal::AbstractTransformationBasedProposal{T},unit_rnd_coords::AbstractVector{T}) where T


Interface function: transform `unit_rnd_coords` into coordinate SVectors according to the given `proposal`.
"""
function _transform! end

# TODO:
# - check if rand and transform are fused on GPU
# - if not: benchmark against local transform
function _rand!(
    rng::AbstractRNG,
    proposal::AbstractTransformationBasedProposal{T},
    dest::AbstractVector{T},
) where {T}
    _rand!(rng, UnitCoordSampler(proposal), dest)
    transform!(proposal, dest)

    return dest
end
