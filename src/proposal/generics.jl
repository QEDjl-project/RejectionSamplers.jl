
# generic fallbacks

# use the default rng for given dest-array and fall back to the interface
function Random.rand!(proposal::AbstractProposal, dest::AbstractVector)
    rng = default_rng(dest)
    _rand!(rng, proposal, dest)
end
