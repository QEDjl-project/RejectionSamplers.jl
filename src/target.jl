# TODO: put these functions in logically distinct files

abstract type AbstractTargetDistribution end
Base.broadcastable(dist::AbstractTargetDistribution) = Ref(dist)

"""

    _compute(dist::AbstractUnivariatTargetDistribution,x::Real)::Real
    _compute(dist::AbstractMultivariateTarget,x::SVector{2,Real})::Real

Return value of `dist` computed at `x`.
"""
function _compute end

# TODO: move this to the abstract event generators
"""

    maximum_value(dist::AbstractTargetDistribution)

Interface function: return the (approximate) maximum_value of a given target distribution.

"""
function maximum_value end

# _compute_kernel(backend, 32)(
#   dest,
#   Base.Fix1(_compute,target_distribution(eg)),
#   batch.args,
#   ndrange=size(dest)
#   )
@kernel inbounds = true function _compute_kernel(dest, @Const(dist), @Const(x))
    I = @index(Global)
    @inbounds dest[I] = _compute(dist, x[I])
end

# TODO: test these function
function compute!(dist::D, dest::AbstractArray, x::AbstractArray) where {D <: AbstractTargetDistribution}
    length(dest) == length(x) || throw(
        ArgumentError(
            "Argument `x` and destination `dest` must have the same length"
        )
    )

    _compute!(dist, dest, x)
    return nothing
end

# TODO: test these function
function compute(dist::AbstractTargetDistribution, x::AbstractVector)
    dist = _allocate_destination(dist, x)
    _compute(dist, dest, x)
    return dist
end

# univariate distribution

abstract type AbstractUnivariatTargetDistribution <: AbstractTargetDistribution end
@inline function _allocate_destination(dist::AbstractUnivariatTargetDistribution, x::AbstractArray{T}) where {T <: Real}
    return similar(x)
end
@inline function _compute!(
        dist::D,
        dest::A,
        x::I,
    ) where {
        D <: AbstractUnivariatTargetDistribution,
        T <: Real,
        A <: AbstractVector{T},
        I <: AbstractVector{T},
    }

    backend = get_backend(dest)
    _compute_kernel(backend, 32)(
        dest,
        dist,
        x,
        ndrange = size(dest)
    )

    return nothing
end

# multivariate distributions

abstract type AbstractMultivariateTarget{N} <: AbstractTargetDistribution end
@inline function _allocate_destination(dist::AbstractMultivariateTarget, x::AbstractArray{TT}) where {N, T <: Real, TT <: SVector{N, T}}
    return allocate(get_backend(x), T, (length(x),))
end
@inline function _compute!(
        dist::D,
        dest::A,
        x::I,
    ) where {
        N,
        D <: AbstractMultivariateTarget{N},
        T <: Real,
        TT <: SVector{N, T},
        A <: AbstractVector{T},
        I <: AbstractVector{TT},
    }

    backend = get_backend(dest)
    _compute_kernel(backend, 32)(
        dest,
        dist,
        x,
        ndrange = size(dest)
    )

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
