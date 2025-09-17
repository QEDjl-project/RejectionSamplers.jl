# TODO:
# - implement/copy UnivariateUniformProposal
# - implement/copy MultivariateUniformProposal
# - implement both versions: local and global transform proposal


### uniform univariate proposal

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

# TODO: implement Proposal interface, not Distributions._rand!
# Random Interface (important for GPU)
function _rand!(
        rng::AbstractRNG,
        d::UniformUnivariateProposal{T},
        A::AbstractArray{T},
    ) where {T <: Real}

    # fallback on Distributions.Uniform (which works on GPU)
    return rand!(rng, d.dist, A)
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

# global version

struct GlobalTransformUniformProposal{T, N} <: AbstractGlobalTransformProposal{SVector{N, T}}
    lower::NTuple{N, T}
    upper::NTuple{N, T}
    function GlobalTransformUniformProposal(
            lower::NTuple{N, T},
            upper::NTuple{N, T},
        ) where {T, N}
        _assert_correct_boundaries(lower, upper)
        return new{T, N}(lower, upper)
    end
end


GlobalTransformUniformProposal(lower::AbstractVector, upper::AbstractVector) =
    GlobalTransformUniformProposal(Tuple(lower), Tuple(upper))

degrees_of_freedom(::GlobalTransformUniformProposal{T, N}) where {T, N} = N

Base.extrema(p::GlobalTransformUniformProposal) = (minimum(p), maximum(p))
Base.minimum(p::GlobalTransformUniformProposal) = p.lower
Base.maximum(p::GlobalTransformUniformProposal) = p.upper
Base.eltype(::GlobalTransformUniformProposal{T, N}) where {T, N} = SVector{N, T}

function _global_transform!(
        proposal::GlobalTransformUniformProposal{T, N},
        A::AbstractVector{TT},
    ) where {T, N, TT <: SVector{N, T}}
    A .= _transform_uniform_val.(A, Ref(proposal.lower), Ref(proposal.upper))
    return A
end

# local version

struct LocalTransformUniformProposal{T, N} <: AbstractLocalTransformProposal{SVector{N, T}}
    lower::NTuple{N, T}
    upper::NTuple{N, T}
    function LocalTransformUniformProposal(
            lower::NTuple{N, T},
            upper::NTuple{N, T},
        ) where {T, N}
        _assert_correct_boundaries(lower, upper)
        return new{T, N}(lower, upper)
    end
end

# connection to the old implementation
# TODO: remove after tests are adjusted
const UniformMultivariateProposal{T, N} = LocalTransformUniformProposal{T, N} where {T, N}

LocalTransformUniformProposal(lower::AbstractVector, upper::AbstractVector) =
    LocalTransformUniformProposal(Tuple(lower), Tuple(upper))

degrees_of_freedom(::LocalTransformUniformProposal{T, N}) where {T, N} = N

Base.extrema(p::LocalTransformUniformProposal) = (minimum(p), maximum(p))
Base.minimum(p::LocalTransformUniformProposal) = p.lower
Base.maximum(p::LocalTransformUniformProposal) = p.upper
Base.eltype(::LocalTransformUniformProposal{T, N}) where {T, N} = SVector{N, T}

function _local_transform(
        proposal::LocalTransformUniformProposal{T, N},
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
    )
end
