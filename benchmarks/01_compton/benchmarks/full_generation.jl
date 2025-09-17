mod = PerturbativeQED()
@info "used model: $mod"
psl = ComptonSphericalLayout(ComptonRestSystem())
@info "used psl: $psl"

nevent_vec = Int.(2.0 .^ (5:6))
@info "used nevents: $nevent_vec"
batch_size_vec = Int.(2.0 .^ (5:6))
@info "used batch sizes: $batch_size_vec"

group = addgroup!(SUITE, "full_generation")
for dtype in DTYPES

    local _group = addgroup!(group, "$dtype")
    arg_type = SVector{3, dtype}

    dist, proposal, max_value = generation_setup(RNG, arg_type, dtype, mod, psl)

    @info "Adding benchmark problems"
    for N in nevent_vec
        _group[N] = BenchmarkGroup()
        for batch_size in batch_size_vec
            bs = min(batch_size, N)

            _group[bs][N] = @benchmarkable begin
                GPUEventGenerators.generate_events(
                    $dist,
                    $proposal,
                    $max_value,
                    $N,
                    $bs,
                    $BACKEND,
                    $dtype,
                    $arg_type,
                )
                KernelAbstractions.synchronize($BACKEND)
            end
        end
    end
end
