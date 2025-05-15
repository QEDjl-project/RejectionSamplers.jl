
RNG = Xoshiro(137137)
N = 256

@testset "proposal" begin
    @testset "$VECTOR_T" for VECTOR_T in VECTOR_TYPES
        @testset "$T" for T in FLOAT_TYPES[VECTOR_T]
            @testset "Uniform proposal" begin
                MIN = -rand(RNG, T)
                MAX = rand(RNG, T)
                u = UniformProposal(MIN, MAX)


                @testset "properties" begin
                    @test minimum(u) == MIN
                    @test maximum(u) == MAX
                    @test extrema(u) == (MIN, MAX)
                end

                @testset "reproducability" begin
                    d_payload1 = VECTOR_T(zeros(T, N))
                    d_payload2 = VECTOR_T(zeros(T, N))

                    Random.seed!(137)
                    rand!(u, d_payload1)

                    Random.seed!(137)
                    rand!(u, d_payload2)

                    @test all(Array(d_payload1) .== Array(d_payload2))
                end

                @testset "sanitiy checks" begin
                    d_payload = VECTOR_T(zeros(T, N))


                    Random.seed!(137)
                    rand!(u, d_payload)

                    @test !any(iszero.(Array(d_payload)))

                    Random.seed!(137)
                    rand!(u, d_payload)

                    @test !any(iszero.(Array(d_payload)))

                end
            end
        end
    end
end
