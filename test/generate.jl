RNG = Xoshiro(137137)
N = Int(2^10)
batch_size = Int(2^10)


@testset "event generation" begin
    @testset "$VECTOR_T" for VECTOR_T in VECTOR_TYPES
        @testset "$T" for T in FLOAT_TYPES[VECTOR_T]

            mu = 5 * rand(RNG, T)                  # central value
            sig = rand(RNG, T)                     # variance
            dom = T.((mu - 0.5 * sig, mu + 3.0 * sig))   # support/domain

            ## example distribution to be sampled
            dist = TruncatedGaussian1D(mu, sig, dom)
            proposal = UniformProposal(dom)

            ## max value
            max_value = maximum_value(dist)

            # TODO: find better way to get backend
            BACKEND = get_backend(VECTOR_T([1.0f0]))
            data = GPUEventGenerators.generate_events(
                dist,
                proposal,
                max_value,
                N,
                batch_size,
                BACKEND,
                T,
            )

            @testset "shape and type preservation" begin
                @test size(data[1]) == (N,)
                @test size(data[2]) == (N,)
                @test eltype(data[1]) == T # for TruncatedGaussian1D
                @test eltype(data[2]) == T
            end

        end
    end
end
