mod = PerturbativeQED()
@info "used model: $mod"
psl = ComptonSphericalLayout(ComptonRestSystem())
@info "used psl: $psl"

nevent_vec = Int.(2.0 .^ (5:6))
@info "used nevents: $nevent_vec"
batch_size_vec = Int.(2.0 .^ (5:6))
@info "used batch sizes: $batch_size_vec"

group = addgroup!(SUITE, "hotloop")
for dtype in DTYPES

    local _group = addgroup!(group, "$dtype")
    arg_type = SVector{3, dtype}

    dist, proposal, max_value = generation_setup(RNG, arg_type, dtype, mod, psl)

    for N in nevent_vec
        _group[N] = BenchmarkGroup()
        for batch_size in batch_size_vec
            bs = min(batch_size, N)

            _group[bs][N] = @benchmarkable @sb(
                begin
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
                end
            ) setup = begin

                # allocate input buffer on GPU (batch_size)
                d_args = allocate($BACKEND, $arg_type, ($batch_size,))
                d_probs = allocate($BACKEND, $dtype, ($batch_size,))
                d_vals = allocate($BACKEND, $dtype, ($batch_size,))

                # allocate output buffer on GPU (out_size = res_size + batch_size)
                out_size = $N + $batch_size
                d_out_args = allocate($BACKEND, $arg_type, (out_size,))
                d_out_vals = allocate($BACKEND, $dtype, (out_size,))

                # allocate current_out_size + init with zero
                current_out_size = KernelAbstractions.zeros($BACKEND, UInt32, 1)

                # trigger for hot loop
                running = true
            end
        end
    end
end
