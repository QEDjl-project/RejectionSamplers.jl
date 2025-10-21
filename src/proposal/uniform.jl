### uniform univariate proposal
# - only for testing purposes
# - use UnifromProposal instead

struct UniformUnivariateProposal{T <: Real, DIST} <: AbstractUnivariateProposal{T}
    dist::DIST

    function UniformUnivariateProposal(a::T, b::T) where {T <: Real}
        dist = Uniform(a, b)
        return new{T, typeof(dist)}(dist)
    end
end
UniformUnivariateProposal(d::NTuple{2}) = UniformUnivariateProposal(d...)

Base.extrema(p::UniformUnivariateProposal) = extrema(p.dist)
Base.minimum(p::UniformUnivariateProposal) = minimum(p.dist)
Base.maximum(p::UniformUnivariateProposal) = maximum(p.dist)
Base.eltype(::UniformUnivariateProposal{T}) where {T} = T

function _propose!(
        rng::AbstractRNG,
        proposal::UniformUnivariateProposal{T},
        sample_dest::AbstractVector,
        weight_dest::AbstractVector,
        backend::KernelAbstractions.Backend
    ) where {T}

    # fallback on Distributions.Uniform (which works on GPU)
    rand!(rng, proposal.dist, sample_dest)
    return fill!(weight_dest, one(T))
end

### uniform multivariate proposal

# generic transformation of x in (0,1) to (low,high)
_transform_uniform_val(x::Real, low::Real, high::Real) = (high - low) * x + low
_transform_uniform_val(x::SVector, low::Tuple, high::Tuple) =
    _transform_uniform_val.(x, low, high)

function _assert_correct_boundaries(::Tuple{}, ::Tuple{}) end

function _assert_correct_boundaries(
        lower::Tuple{Vararg{T, N}},
        upper::Tuple{Vararg{T, N}},
    ) where {T <: Real, N}
    first(lower) <= first(upper) || throw(
        ArgumentError(
            "lower boundary need to be smaller or equal to the respective upper boundary",
        ),
    )
    return _assert_correct_boundaries(lower[2:end], upper[2:end])
end

# local version

# TODO: extent using Ts and Tw to allow NTuple *and* SVector
struct UniformProposal{T, N} <: AbstractTransformProposal{SVector{N, T}, T}
    lower::NTuple{N, T}
    upper::NTuple{N, T}
    function UniformProposal(
            lower::NTuple{N, T},
            upper::NTuple{N, T},
        ) where {T, N}
        _assert_correct_boundaries(lower, upper)
        return new{T, N}(lower, upper)
    end
end

# connection to the old implementation
# TODO: remove after tests are adjusted
const UniformMultivariateProposal{T, N} = UniformProposal{T, N} where {T, N}

UniformProposal(lower::AbstractVector, upper::AbstractVector) =
    UniformProposal(Tuple(lower), Tuple(upper))

degrees_of_freedom(::UniformProposal{T, N}) where {T, N} = N

Base.extrema(p::UniformProposal) = (minimum(p), maximum(p))
Base.minimum(p::UniformProposal) = p.lower
Base.maximum(p::UniformProposal) = p.upper

function _transform(
        proposal::UniformProposal{T, N},
        v::SVector{N, T},
    ) where {T, N}

    return SVector{N, T}(
            ntuple(
                x -> _transform_uniform_val(
                    getindex(v, x),
                    getindex(proposal.lower, x),
                    getindex(proposal.upper, x),
                ),
                N,
            ),
        ), one(T)
end
