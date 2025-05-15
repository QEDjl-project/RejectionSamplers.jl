
abstract type AbstractTargetDistribution end
Base.broadcastable(dist::AbstractTargetDistribution) = Ref(dist)

"""

    _compute!(dist,dest,x)

Computes the target distribution `dist` on `x` and stores the result in `dest`.
"""
function _compute! end

abstract type AbstractUnivariatTargetDistribution <: AbstractTargetDistribution end

"""

    _compute(dist,x::AbstractUnivariatTargetDistribution)::Real

Return value of `dist` computed at `x`.
"""
function _compute end

"""

    maximum_value(dist::AbstractTargetDistribution)

Interface function: return the (approximate) maximum_value of a given target distribution.

"""
function maximum_value end

@inline function _compute!(
    dist::D,
    dest::A,
    x::A,
) where {D<:AbstractUnivariatTargetDistribution,T<:Real,A<:AbstractArray{T}}

    broadcast!(Base.Fix1(_compute, dist), dest, x)

    return nothing
end

# generic fallback - calculates on host and copies to device
#
# note:
# - this allocates on host
#
function _compute!(
    dist::D,
    dest::AbstractArray{T},
    x::AbstractArray{T},
) where {D<:AbstractUnivariatTargetDistribution,T<:Real}

    copyto!(dest, broadcast(Base.Fix1(_compute, dist), x))
end
