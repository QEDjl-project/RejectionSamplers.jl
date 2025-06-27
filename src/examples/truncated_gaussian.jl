
module TruncatedGaussians

using Distributions
using StatsFuns
using Random
using QuadGK

using GPUEventGenerators

export TruncatedGaussian1D
export TruncatedGaussian

# test target distribution

struct TruncatedGaussian1D{T<:Real} <:
       GPUEventGenerators.AbstractUnivariatTargetDistribution
    mu::T
    sig::T
    lower::T
    upper::T

    function TruncatedGaussian1D(mu::T, sig::T, lower::T, upper::T) where {T<:Real}

        @assert lower < upper
        @assert sig > zero(sig)

        return new{T}(mu, sig, lower, upper)
    end
end

function TruncatedGaussian1D(mu::T, sig::T, domain::NTuple{2,T}) where {T<:Real}
    return TruncatedGaussian1D(mu, sig, domain...)
end

endpoints(dist::TruncatedGaussian1D) = (dist.lower, dist.upper)
is_in_domain(d::TruncatedGaussian1D, x) = d.lower <= x <= d.upper

function GPUEventGenerators.maximum_value(d::TruncatedGaussian1D{T}) where {T}
    crit_pt = min(max(d.mu, d.lower), d.upper)
    return _unsafe_compute(d, crit_pt)
end

_unsafe_compute(d::TruncatedGaussian1D, x) = normpdf(d.mu, d.sig, x)
GPUEventGenerators._compute(d::TruncatedGaussian1D{T}, x) where {T} =
    is_in_domain(d, x) ? _unsafe_compute(d, x) : zero(T)


### multivariate version

# checks if lower < upper for all dims

function _assert_correct_boundaries(
    lower::NTuple{N,T},
    upper::NTuple{N,T},
) where {T<:Real,N}
    for i = 1:N
        lower[i] <= upper[i] || throw(
            ArgumentError(
                """
                lower boundary need to be smaller or equal to the respective upper boundary, but\n
                lower[$i] = $(lower[i]) > $(upper[i]) = upper[$i]
                """,
            ),
        )
    end
    return nothing
end

struct TruncatedGaussian{T<:Real,N} <: GPUEventGenerators.AbstractMultivariateTarget{N}

    mu::NTuple{N,T}
    sig::NTuple{N,T}
    lower::NTuple{N,T}
    upper::NTuple{N,T}

    function TruncatedGaussian(
        mu::TT,
        sig::TT,
        lower::TT,
        upper::TT,
    ) where {N,T<:Real,TT<:NTuple{N,T}}
        _assert_correct_boundaries(lower, upper)
        return new{T,N}(mu, sig, lower, upper)
    end
end

endpoints(d::TruncatedGaussian) = (d.lower, d.upper)

function is_in_domain(d::TruncatedGaussian{T,N}, x::NTuple{N,T}) where {N,T}
    return all(d.lower .<= x .<= d.upper)
end



function GPUEventGenerators.maximum_value(d::TruncatedGaussian{T,N}) where {T,N}
    crit_pt = @. min(max(d.mu, d.lower), d.upper)

    return _unsafe_compute(d, crit_pt)
end

_unsafe_compute(d::TruncatedGaussian{T,N}, x) where {N,T} =
    prod(i -> normpdf(d.mu[i], d.sig[i], x[i]), N)
GPUEventGenerators._compute(d::TruncatedGaussian{T}, x) where {T} =
    is_in_domain(d, x) ? _unsafe_compute(d, x) : zero(T)


end
