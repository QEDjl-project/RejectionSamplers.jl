PARAMETER_RNG = Xoshiro(137137)
test_default_rng(::Type{<:AbstractArray}) = Random.default_rng()
test_default_rng(T::Type{<:GPUArrays.AnyGPUArray}) = GPUArrays.default_rng(T)

function testsuite_proposal(backend, vec_type, el_type, N)
    @testset "Uniform univariate proposal" begin
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

            Random.seed!(137)
            rand!(u, d_payload1)

            Random.seed!(137)
            rand!(u, d_payload2)

            @test all(Array(d_payload1) .== Array(d_payload2))
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
    return @testset "Uniform multivariate proposal" begin

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
                d_payload = vec_type(Vector{NTuple{dim, el_type}}(undef, N))

                RNG = test_default_rng(vec_type)
                rand!(u, d_payload)

                @test eltype(Array(d_payload)) == NTuple{dim, el_type}

            end
        end
    end
end
