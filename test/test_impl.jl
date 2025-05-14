
using StatsFuns

# test target distribution

struct TruncatedGaussian1D{T<:Real} <:
       GPUEventGenerators.AbstractUnivariatTargetDistribution
    mu::T
    sig::T
    low::T
    high::T

    function TruncatedGaussian1D(mu::T, sig::T, low::T, high::T) where {T<:Real}

        @assert low < high
        @assert sig > zero(sig)

        return new{T}(mu, sig, low, high)
    end
end

function TruncatedGaussian1D(mu::T, sig::T, domain::NTuple{2,T}) where {T<:Real}
    return TruncatedGaussian1D(mu, sig, domain...)
end

endpoints(dist::TruncatedGaussian1D) = (dist.low, dist.high)
is_in_domain(d, x) = d.low <= x <= d.high

function maximum_value(d::TruncatedGaussian1D{T}) where {T}
    mu = d.mu

    if is_in_domain(d, mu)
        return _unsafe_compute(d, mu)
    end

    if mu <= d.low
        return _unsafe_compute(d, d.low)
    end

    return _unsafe_compute(d, d.high)
end

_unsafe_compute(d::TruncatedGaussian1D, x) = normpdf(d.mu, d.sig, x)
GPUEventGenerators._compute(d::TruncatedGaussian1D{T}, x) where {T} =
    is_in_domain(d, x) ? _unsafe_compute(d, x) : zero(T)
