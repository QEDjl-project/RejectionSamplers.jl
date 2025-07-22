abstract type AbstractTargetDistribution end
Base.broadcastable(dist::AbstractTargetDistribution) = Ref(dist)

"""

    _compute!(dist,dest,x)

Computes the target distribution `dist` on `x` and stores the result in `dest`.
"""
function _compute! end

abstract type AbstractUnivariatTargetDistribution <: AbstractTargetDistribution end

"""

    _compute(dist::AbstractUnivariatTargetDistribution,x::Real)::Real
    _compute(dist::AbstractMultivariateTarget,x::SVector{2,Real})::Real

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
    ) where {D <: AbstractUnivariatTargetDistribution, T <: Real, A <: AbstractArray{T}}

    broadcast!(Base.Fix1(_compute, dist), dest, x)

    return nothing
end

# TODO: see if this is necessary
# generic fallback - calculates on host and copies to device
#
# note:
# - this allocates on host
#
function _compute!(
        dist::D,
        dest::AbstractArray{T},
        x::AbstractArray{T},
    ) where {D <: AbstractUnivariatTargetDistribution, T <: Real}

    return copyto!(dest, broadcast(Base.Fix1(_compute, dist), x))
end

# bivariate distributions

abstract type AbstractMultivariateTarget{N} <: AbstractTargetDistribution end


@inline function _compute!(
    dist::D,
    dest::A,
    x::I,
) where {
    N,
    D<:AbstractMultivariateTarget{N},
    T<:Real,
    TT<:SVector{N,T},
    A<:AbstractVector{T},
    I<:AbstractVector{TT},
}

    broadcast!(Base.Fix1(_compute, dist), dest, x)

    return nothing
end

# TODO: see if this is necessary
# generic fallback - calculates on host and copies to device
#
# note:
# - this allocates on host
#=
function _compute!(
    dist::D,
    dest::AbstractArray{T},
    x::AbstractArray{TT},
) where {D<:AbstractUnivariatTargetDistribution,T<:Real,TT<:NTuple{2,T}}

    copyto!(dest, broadcast(Base.Fix1(_compute, dist), x))
end
=#


#=
# scattering process distribution

abstract type AbstractScatteringProcessDistribution <: AbstractTargetDistribution end

@inline function _compute!(
    dist::D,
    dest::A,
    x::I,
) where {
        D<:AbstractScatteringProcessDistribution,
        T<:Real,
        A<:AbstractArray{T},
        PSP<:AbstractPhaseSpacePoint,
        I<:AbstractVector{PSP}
    }

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
    dest::AbstractVector{T},
    x::AbstractVector{PSP},
) where {
        D<:AbstractScatteringProcessDistribution,
        T<:Real,
        PSP<:AbstractPhaseSpacePoint,
    }

    copyto!(dest, broadcast(Base.Fix1(_compute, dist), x))
end
=#
