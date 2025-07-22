struct NaiveMaxFinder <: AbstractSampleBasedMaxFinder
    nsamples::Int
end

_nsamples(n::NaiveMaxFinder) = n.nsamples

function _findmax(method::NaiveMaxFinder, weights::AbstractVector)
    return maximum(weights)
end
