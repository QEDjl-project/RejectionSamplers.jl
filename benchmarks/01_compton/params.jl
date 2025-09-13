# WARNING:
# needs to be included into a scope with the right packages loaded

const RNG = Xoshiro(1234)

const MODEL = PerturbativeQED()
@info "used model: $MODEL"
const PSL = ComptonSphericalLayout(ComptonRestSystem())
@info "used psl: $PSL"

const NS = Int.(2.0 .^ (5:6))
const BATCH_SIZES = Int.(2.0 .^ (5:6))

@info "Problem sizes (number of events): $NS"
@info "Batch sizes: $BATCH_SIZES"

function create_parameters(dtype)
    @info "Create parameters"
    om = dtype(2.0e-3) # 1keV
    @info "om = $om"
    lower = dtype.((om, -1.0, 0.0))
    upper = dtype.((om, 1.0, 2 * pi))
    dom = (lower, upper)

    return dom
end
