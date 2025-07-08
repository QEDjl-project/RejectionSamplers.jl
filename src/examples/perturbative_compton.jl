module PerturbativeCompton

export KleinNishinaDistribution

const ALPHA = inv(137.035999074)
const ALPHA32 = inv(137.035999074f0)

get_alpha(::Type{Float64}) = ALPHA
get_alpha(::Type{Float32}) = ALPHA32

using GPUEventGenerators

struct KleinNishinaDistribution{T} <: GPUEventGenerators.AbstractMultivariateTarget{2}
    omega::T

    function KleinNishinaDistribution(omega::T) where {T<:Real}
        omega > zero(omega) || throw(ArgumentError("omega needs to be positive."))
        return new{T}(omega)
    end
end

function _omega_prime(omega, cos_theta)
    return omega / (1 + omega * (1 - cos_theta))
end

function _klein_nishina_formula(omega::T, cos_theta::T, phi::T) where {T<:Real}
    om_prime = _omega_prime(omega, cos_theta)
    omp_over_om = om_prime / omega
    prefac = pi * get_alpha(T)^2
    return prefac *
           omp_over_om^2 *
           (omp_over_om + inv(omp_over_om) - one(omega) + cos_theta^2)
end

function GPUEventGenerators.maximum_value(d::KleinNishinaDistribution{T}) where {T<:Real}
    return _klein_nishina_formula(d.omega, one(T), zero(T))
end

function GPUEventGenerators._compute(
    dist::KleinNishinaDistribution{T},
    cth_phi::NTuple{2,T},
) where {T<:Real}
    return _klein_nishina_formula(dist.omega, cth_phi...)
end


end
