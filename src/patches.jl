

# TODO: move this to QEDcore
function QEDcore.PhaseSpacePoint(
    p::AbstractProcessDefinition,
    m::AbstractModelDefinition,
    psl::AbstractPhaseSpaceLayout,
    coords::Tuple,
)

    in_dim = phase_space_dimension(p, m, in_phase_space_layout(psl))
    out_dim = phase_space_dimension(p, m, psl)
    return PhaseSpacePoint(
        p,
        m,
        psl,
        ntuple(i -> coords[i], in_dim),
        ntuple(i -> coords[in_dim+i], out_dim),
    )
end

QEDcore.momenta(psp::AbstractPhaseSpacePoint) =
    (momenta(psp, Incoming())..., momenta(psp, Outgoing())...)
