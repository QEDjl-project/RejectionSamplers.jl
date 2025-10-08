module HardScattering

export HardScatteringDistribution

using RejectionSamplers
using QEDbase

struct HardScatteringDistribution{
        DOF,
        P <: AbstractProcessDefinition,
        M <: AbstractModelDefinition,
        PSL <: AbstractOutPhaseSpaceLayout,
    } <: RejectionSamplers.AbstractMultivariateTarget{DOF}

    proc::P
    model::M
    psl::PSL

    function HardScatteringDistribution(
            proc::P,
            model::M,
            psl::PSL
        ) where {
            P <: AbstractProcessDefinition,
            M <: AbstractModelDefinition,
            PSL <: AbstractOutPhaseSpaceLayout,
        }

        in_dim = phase_space_dimension(proc, model, in_phase_space_layout(psl))
        out_dim = phase_space_dimension(proc, model, psl)
        DOF = in_dim + out_dim

        return new{P, M, PSL, DOF}(proc, model, psl)
    end
end

# TODO: insert max-finder functionality instead (e.g. from QEDprobing.jl)
function RejectionSamplers.maximum_value(d::HardScatteringDistribution) end

function RejectionSamplers._compute(
        dist::HardScatteringDistribution{DOF},
        coords::NTuple{DOF, T},
    ) where {DOF, T <: Real}

    psp = PhaseSpacePoint(dist.proc, dist.model, dist.psl, coords)

    return differential_cross_section(psp)
end


end
