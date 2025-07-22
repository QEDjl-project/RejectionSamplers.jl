PARAMETER_RNG = Xoshiro(137137)
test_default_rng(::Type{<:AbstractArray}) = Random.default_rng()
test_default_rng(T::Type{<:GPUArrays.AnyGPUArray}) = GPUArrays.default_rng(T)

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
            d_payload1 = vec_type(zeros(el_type, N))
            d_payload2 = vec_type(zeros(el_type, N))
            RNG = GPUEventGenerators.default_rng(typeof(d_payload1))
            RNG2 = deepcopy(RNG)

            GPUEventGenerators._rand!(RNG, u, d_payload1)
            GPUEventGenerators._rand!(RNG, u, d_payload2)

            # FIXME: fix reproducibility for event generation
            @test_broken isapprox(Array(d_payload1), Array(d_payload2))
        end

        @testset "sanity checks" begin
            d_payload = vec_type(zeros(el_type, N))

            # FIXME: rand(RNG,Float16) could return Float16(0.0)
            # see https://discourse.julialang.org/t/output-distribution-of-rand-float32-and-rand-float64-thread-2/105184
            Random.seed!(137)
            rand!(u, d_payload)

            el_type != Float16 ? (@test !any(iszero.(Array(d_payload)))) : nothing
            @test eltype(Array(d_payload)) == el_type

        end
    end
    @testset "local transform multivariate" begin

        DIMS = (1, rand(RNG, 2:4))
        @testset "dim = $dim" for dim in DIMS

            MIN = Tuple(-rand(PARAMETER_RNG, el_type, dim))
            MAX = Tuple(rand(PARAMETER_RNG, el_type, dim))
            u = UniformMultivariateProposal(MIN, MAX)

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
                d_payload = vec_type(Vector{SVector{dim,el_type}}(undef, N))

                RNG = test_default_rng(vec_type)
                rand!(u, d_payload)

                @test eltype(Array(d_payload)) == SVector{dim,el_type}

            end
        end
    end

    @testset "global transform multivariate" begin

        DIMS = (1, rand(RNG, 2:4))
        @testset "dim = $dim" for dim in DIMS

            MIN = Tuple(-rand(PARAMETER_RNG, el_type, dim))
            MAX = Tuple(rand(PARAMETER_RNG, el_type, dim))
            u = GlobalTransformUniformProposal(MIN, MAX)

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
                d_payload = vec_type(Vector{SVector{dim,el_type}}(undef, N))

                RNG = test_default_rng(vec_type)
                rand!(u, d_payload)

                @test eltype(Array(d_payload)) == SVector{dim,el_type}

            end
        end
    end
end
