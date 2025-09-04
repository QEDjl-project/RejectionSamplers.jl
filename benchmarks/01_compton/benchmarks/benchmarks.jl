# WARNING:
# needs to be included into a scope with the right packages loaded

function benchmark_full_generation(rng, backend, dtype, mod, psl, nevent_vec, batch_size_vec)
    instance_suite = BenchmarkGroup()
    @info "building benchmark with backend=$backend, dtype=$dtype"

    dom = create_parameters(dtype)

    arg_type = SVector{3, dtype}

    dist = ComptonDistribution{dtype}(mod, psl)
    proposal = UniformMultivariateProposal(dom...)

    @info "Finding maximum ..."
    max_value = findmax(
        rng,
        dtype,
        dist,
        proposal,
        QuantileReductionMethod(dtype(0.001), Int(1.0e6));
        dtype = arg_type,
    )
    @info "found at $max_value with type $(typeof(max_value))"

    @info "Adding benchmark problems"
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

function benchmark_generation(rng, backend, dtype, mod, psl, nevent_vec, batch_size_vec)
    instance_suite = BenchmarkGroup()
    @info "building benchmark with backend=$backend, dtype=$dtype"

    dom = create_parameters(dtype)

    arg_type = SVector{3, dtype}

    dist = ComptonDistribution{dtype}(mod, psl)
    proposal = UniformMultivariateProposal(dom...)

    @info "Finding maximum ..."
    max_value = findmax(
        rng,
        dtype,
        dist,
        proposal,
        QuantileReductionMethod(dtype(0.001), Int(1.0e6));
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
                while running
                    @inline GPUEventGenerators._generate_events!(
                        $dist,
                        $proposal,
                        $max_value,
                        d_args,
                        d_probs,
                        d_vals,
                        d_out_args,
                        d_out_vals,
                        current_out_size,
                    )

                    h_current_out_size = Vector(current_out_size)[1]
                    if h_current_out_size >= $N
                        running = false
                    end
                end
                KernelAbstractions.synchronize($backend)
                # free allocations
            end setup = begin
                # allocate input buffer on GPU (batch_size)
                d_args = allocate($backend, $arg_type, ($batch_size,))
                d_probs = allocate($backend, $dtype, ($batch_size,))
                d_vals = allocate($backend, $dtype, ($batch_size,))

                # allocate output buffer on GPU (out_size = res_size + batch_size)
                out_size = $N + $batch_size
                d_out_args = allocate($backend, $arg_type, (out_size,))
                d_out_vals = allocate($backend, $dtype, (out_size,))

                # allocate current_out_size + init with zero
                current_out_size = KernelAbstractions.zeros($backend, UInt32, 1)

                # trigger for hot loop
                running = true
            end
        end
    end
    return instance_suite
end
