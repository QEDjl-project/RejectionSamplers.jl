# TODO: build own benchmark suite for buffer
# proposal are somewhat independent from the processes

bench_name = "proposal"
@info "Adding benchmark: $bench_name"

batch_size_vec = 2 .^ (5:6)
@info "used batch sizes: $batch_size_vec"

group = addgroup!(SUITE, bench_name)
group = addgroup!(group, "uniform")
for out_type in DTYPES

    local _group = addgroup!(group, "$out_type")
    in_type = SVector{3, out_type}

    dom = create_parameters(out_type)
    proposal = UniformMultivariateProposal(dom...)

    for batch_size in batch_size_vec

        _group[batch_size] = @benchmarkable @sb(
            begin
                GPUEventGenerators.generate_proposals!($proposal, batch_buffer)
            end
        ) setup = begin
            reclaim_mem()
            batch_buffer = GPUEventGenerators._allocate_batch_buffer($BACKEND, $in_type, $out_type, $batch_size)
        end
    end
end
