bench_name = "full"
@info "Adding benchmark: $bench_name"
mod = PerturbativeQED()
@info "used model: $mod"
psl = ComptonSphericalLayout(ComptonRestSystem())
@info "used psl: $psl"

nevent_vec = 2 .^ (5:6)
@info "used nevents: $nevent_vec"
batch_size_vec = 2 .^ (5:6)
@info "used batch sizes: $batch_size_vec"

group = addgroup!(SUITE, bench_name)
for dtype in DTYPES

    local _group = addgroup!(group, "$dtype")
    arg_type = SVector{3, dtype}

    dist, proposal, max_value = generation_setup(RNG, arg_type, dtype, mod, psl)

    @info "Adding benchmark problems"
    for N in nevent_vec
        _group[N] = BenchmarkGroup()
        for batch_size in batch_size_vec
            if batch_size <= N

                _group[batch_size][N] = @benchmarkable begin
                    GPUEventGenerators.generate_events(
                        $dist,
                        $proposal,
                        $max_value,
                        $N,
                        $batch_size,
                        $BACKEND,
                        $dtype,
                        $arg_type,
                    )
                    KernelAbstractions.synchronize($BACKEND)
                end
            end
        end
    end
end
