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
for out_type in DTYPES

    local _group = addgroup!(group, "$out_type")
    in_type = SVector{3, out_type}

    dist, proposal, max_value = generation_setup(RNG, in_type, out_type, mod, psl)

    @info "Adding benchmark problems"
    for N in nevent_vec
        _group[N] = BenchmarkGroup()
        for batch_size in batch_size_vec
            if batch_size <= N

                _group[batch_size][N] = @benchmarkable @sb(
                    begin
                        RejectionSamplers.generate_events(
                            $dist,
                            $proposal,
                            $max_value,
                            $N,
                            $batch_size,
                            $BACKEND,
                            $out_type,
                            $in_type,
                        )
                    end
                )
            end
        end
    end
end
