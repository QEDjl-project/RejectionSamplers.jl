function Base.findmax(
        rng::AbstractRNG,
        out_dtype::Type{T},
        target::AbstractTargetDistribution,
        proposal::AbstractProposalDistribution,
        method::AbstractSampleBasedMaxFinder;
        dtype = out_dtype,
    ) where {T <: Real}

    # Stages:
    # - build samples (momenta)
    # - build psps
    # - calculate weights==dcs
    # - perform maxfinder on weight array

    N = _nsamples(method)

    coords = Vector{dtype}(undef, N)
    _rand!(rng, proposal, coords)
    weights = _compute.(target, coords)

    return _findmax(method, weights)

end
