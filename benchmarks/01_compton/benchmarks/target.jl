bench_name = "full"
@info "Adding benchmark: $bench_name"
mod = PerturbativeQED()
@info "used model: $mod"
psl = ComptonSphericalLayout(ComptonRestSystem())
@info "used psl: $psl"

batch_size_vec = 2 .^ (5:6)
@info "used batch sizes: $batch_size_vec"

group = addgroup!(SUITE, bench_name)
for out_type in DTYPES

    local _group = addgroup!(group, "$out_type")
    in_type = SVector{3, out_type}

    dom = create_parameters(out_type)
    target = ComptonDistribution{out_type}(mod, psl)
    proposal = UniformMultivariateProposal(dom...)

    @info "Adding benchmark problems"
    for batch_size in batch_size_vec

        _group[batch_size] = @benchmarkable @sb(
            begin
                RejectionSamplers.evaluate_target!($target, batch_buffer)
            end
        ) setup = begin
            reclaim_mem()
            batch_buffer = RejectionSamplers._allocate_batch_buffer($BACKEND, $in_type, $out_type, $batch_size)
            RejectionSamplers.generate_proposals!($proposal, batch_buffer)
        end
    end
end
