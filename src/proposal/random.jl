# generic fallback to Random.rand! (used on CPU)
_rand!(rng::AbstractRNG, a::Vector) = Random.rand!(rng, a)

"""

    default_rng(v::AbstractVector)

Return default rng to randomize given vector. Uses fallbacks on `Random.default_rng` and `GPUArrays.default_rng`.
"""
function default_rng end

default_rng(::Type{T}) where {T} = Random.default_rng()
default_rng(::Type{V}) where {V <: GPUArrays.AnyGPUArray} = GPUArrays.default_rng(V)
