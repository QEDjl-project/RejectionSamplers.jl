function generation_setup(rng, arg_type, dtype, mod, psl)
    dom = create_parameters(dtype)
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

    return dist, proposal, max_value
end

function create_parameters(dtype)
    @info "Create parameters"
    om = dtype(2.0e-3) # 1keV
    @info "om = $om"
    lower = dtype.((om, -1.0, 0.0))
    upper = dtype.((om, 1.0, 2 * pi))
    dom = (lower, upper)

    return dom
end
