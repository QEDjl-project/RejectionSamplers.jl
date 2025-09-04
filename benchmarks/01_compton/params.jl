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
