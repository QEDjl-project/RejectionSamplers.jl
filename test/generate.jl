RNG = Xoshiro(137137)

function testsuite_event_generation(backend, vec_type, el_type, N, batch_size)
    mu = 5 * rand(RNG, el_type)                  # central value
    sig = rand(RNG, el_type)                     # variance
    dom = el_type.((mu - 0.5 * sig, mu + 3.0 * sig))   # support/domain

    ## example distribution to be sampled
    dist = TruncatedGaussian1D(mu, sig, dom)
    proposal = UniformUnivariateProposal(dom)

    ## max value
    max_value = maximum_value(dist)

    data = GPUEventGenerators.generate_events(
        dist,
        proposal,
        max_value,
        N,
        batch_size,
        backend,
        el_type,
    )

    @testset "shape and type preservation" begin
        @test size(data[1]) == (N,)
        @test size(data[2]) == (N,)
        @test eltype(data[1]) == el_type # for TruncatedGaussian1D
        @test eltype(data[2]) == el_type
    end
end
