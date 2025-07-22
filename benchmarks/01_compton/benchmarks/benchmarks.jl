
# WARNING:
# needs to be included into a scope with the right packages loaded

function benchmark_generation(rng, backend, dtype, mod, psl, nevent_vec, batch_size_vec)
    instance_suite = BenchmarkGroup()
    @info "building benchmark with backend=$backend, dtype=$dtype"

    dom = create_parameters(dtype)

    arg_type = SVector{3,dtype}

    dist = ComptonDistribution{dtype}(mod, psl)
    proposal = UniformMultivariateProposal(dom...)

    @info "Finding maximum ..."
    max_value = findmax(
        rng,
        dtype,
        dist,
        proposal,
        QuantileReductionMethod(dtype(0.001), Int(1e6));
        dtype = arg_type,
    )
    @info "found at $max_value with type $(typeof(max_value))"

    @info "Adding benchmark problems"
    @show batch_size_vec
    for N in nevent_vec
        instance_suite[N] = BenchmarkGroup()
        for batch_size in batch_size_vec
            bs = min(batch_size, N)

            instance_suite[N][bs] = @benchmarkable begin
                GPUEventGenerators.generate_events(
                    $dist,
                    $proposal,
                    $max_value,
                    $N,
                    $bs,
                    $backend,
                    $dtype,
                    $arg_type,
                )
                KernelAbstractions.synchronize($backend)
            end
        end
    end
    return instance_suite
end
