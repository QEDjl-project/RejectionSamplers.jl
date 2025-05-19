
RNG = Xoshiro(137137)

function testsuite_proposal(backend, vec_type, el_type, N)
    @testset "Uniform proposal" begin
        MIN = -rand(RNG, el_type)
        MAX = rand(RNG, el_type)
        u = UniformProposal(MIN, MAX)


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
end
