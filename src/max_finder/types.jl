abstract type AbstractMaxFinder end
abstract type AbstractSampleBasedMaxFinder <: AbstractMaxFinder end

"""

    _nsamples(method::AbstractSampleBasedMaxFinder)

Interface function for sample-based max-finder. Returns the number of samples to be generated for max-finding.
"""
function _nsamples end

"""

    _findmax(method::AbstractSampleBasedMaxFinder, weights::AbstractVector{T})::T where {T<:Real}

Interface function for sample based max-finder. Return the maximum weight of a given vector `weights` according to `method`.
"""
function _findmax end
