module PerturbativeCompton


using GPUEventGenerators
using QEDprocesses
using QEDcore


export KleinNishinaDistribution
export ComptonDistribution

const ALPHA = inv(137.035999074)
const ALPHA32 = inv(137.035999074f0)

get_alpha(::Type{Float64}) = ALPHA
get_alpha(::Type{Float32}) = ALPHA32


### Klein-Nishina

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


### General perturbative Compton scattering

# - just for testing
# - assumes 1D in_psl with coordinate `omega`
# - will be generalized using `ScatteringProcessDistribution`
struct ComptonDistribution{
    T,
    N,
    C<:Compton,
    M<:AbstractModelDefinition,
    PSL<:AbstractOutPhaseSpaceLayout,
} <: GPUEventGenerators.AbstractMultivariateTarget{N}
    proc::C
    model::M
    psl::PSL

    function ComptonDistribution{T}(
        model::M,
        psl::PSL;
        spin_pol = (AllSpin(), AllPol(), AllSpin(), AllPol()),
    ) where {T<:Real,M<:AbstractModelDefinition,PSL<:AbstractOutPhaseSpaceLayout}
        proc = Compton(spin_pol...)
        in_dim = phase_space_dimension(proc, model, in_phase_space_layout(psl))
        out_dim = phase_space_dimension(proc, model, psl)
        DOF = in_dim + out_dim
        return new{T,DOF,typeof(proc),M,PSL}(proc, model, psl)
    end

end

ComptonDistribution(
    model::AbstractModelDefinition,
    psl::AbstractOutPhaseSpaceLayout,
    spin_pol::Tuple,
) = ComptonDistribution{Float64}(model, psl, spin_pol)

function GPUEventGenerators._compute(
    dist::ComptonDistribution{T,DOF},
    coords::NTuple{DOF,T},
) where {DOF,T<:Real}

    psp = PhaseSpacePoint(dist.proc, dist.model, dist.psl, coords)

    return differential_cross_section(psp)
end

end
