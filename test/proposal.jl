PARAMETER_RNG = Xoshiro(137137)

function testsuite_uniform_proposal(backend, vec_type, el_type, N)
    @testset "univariate proposal" begin
        MIN = -rand(PARAMETER_RNG, el_type)
        MAX = rand(PARAMETER_RNG, el_type)
        u = UniformUnivariateProposal(MIN, MAX)


        @testset "properties" begin
            @test minimum(u) == MIN
            @test maximum(u) == MAX
            @test extrema(u) == (MIN, MAX)
        end

        @testset "reproducibility" begin
            d_samples1 = vec_type(zeros(el_type, N))
            d_weights1 = vec_type(zeros(el_type, N))
            d_samples2 = vec_type(zeros(el_type, N))
            d_weights2 = vec_type(zeros(el_type, N))

            RNG = RejectionSamplers.default_rng(typeof(d_samples1))
            RNG2 = deepcopy(RNG)

            propose!(RNG, u, d_samples1, d_weights1; backend)
            propose!(RNG2, u, d_samples2, d_weights2; backend)

            # FIXME: fix reproducibility for event generation
            @test_skip isapprox(Array(d_samples1), Array(d_samples2))
            @test isapprox(Array(d_weights1), Array(d_weights2))
        end

        @testset "sanity checks" begin
            d_samples = vec_type(zeros(el_type, N))
            d_weights = vec_type(zeros(el_type, N))
            RNG = RejectionSamplers.default_rng(typeof(d_samples))

            # FIXME: rand(RNG,Float16) could return Float16(0.0)
            # see https://discourse.julialang.org/t/output-distribution-of-rand-float32-and-rand-float64-thread-2/105184
            propose!(RNG, u, d_samples, d_weights; backend)

            el_type != Float16 ? (@test !any(iszero.(Array(d_samples)))) : nothing
            @test eltype(Array(d_samples)) == el_type
            @test eltype(Array(d_weights)) == el_type
            @test all(isone.(Array(d_weights)))
        end
    end
    return @testset "transform based uniform" begin

        DIMS = (1, rand(RNG, 2:4))
        @testset "dim = $dim" for dim in DIMS

            MIN = Tuple(-rand(PARAMETER_RNG, el_type, dim))
            MAX = Tuple(rand(PARAMETER_RNG, el_type, dim))
            u = UniformProposal(MIN, MAX)

            @testset "properties" begin
                @test minimum(u) == MIN
                @test maximum(u) == MAX
                @test extrema(u) == (MIN, MAX)
            end

            #=
            # FIXME: enable this, after we have a reproducable RNG
            @testset "reproducibility" begin
                d_payload1 = vec_type(Vector{NTuple{dim,el_type}}(undef, N))
                d_payload2 = vec_type(Vector{NTuple{dim,el_type}}(undef, N))

                RNG = test_default_rng(vec_type)
                RNG2 = deepcopy(RNG)
                rand!(RNG,u, d_payload1)
                rand!(RNG2,u, d_payload2)

                @test all(Array(d_payload1) .== Array(d_payload2))
            end
            =#

            @testset "sanity checks" begin
                d_samples = vec_type(Vector{SVector{dim, el_type}}(undef, N))
                d_weights = vec_type(Vector{el_type}(undef, N))

                RNG = RejectionSamplers.default_rng(vec_type)
                propose!(RNG, u, d_samples, d_weights; backend)

                @test eltype(Array(d_samples)) == SVector{dim, el_type}
                @test all(isone.(Array(d_weights)))

            end
        end
    end
end
