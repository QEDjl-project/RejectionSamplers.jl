bench_name = "full"
@info "profiling: $bench_name"
mod = PerturbativeQED()
@info "used model: $mod"
psl = ComptonSphericalLayout(ComptonRestSystem())
@info "used psl: $psl"

# make it so it's definitely only running once
N = 1
@info "used nevents: $N"
batch_size = 2^18
@info "used batch size: $batch_size"
dtype = Float32
@info "used type: $dtype"

arg_type = SVector{3, dtype}

dist, proposal, max_value = generation_setup(RNG, arg_type, dtype, mod, psl)

GPUEventGenerators.generate_events(
    dist,
    proposal,
    max_value,
    N,
    batch_size,
    BACKEND,
    dtype,
    arg_type,
)
KernelAbstractions.synchronize(BACKEND)
