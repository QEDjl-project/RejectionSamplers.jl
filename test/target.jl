
RNG = Xoshiro(137137)

struct TestUnivariateTarget <: GPUEventGenerators.AbstractUnivariatTargetDistribution end
GPUEventGenerators._compute(::TestUnivariateTarget, x) = cos(x)

const UNI_DIST = TestUnivariateTarget()

function testsuite_univariate_target(backend, vec_type, el_type, N)
    d_input = vec_type(rand(RNG, el_type, N))
    d_vals = similar(d_input)

    h_groundtruth = GPUEventGenerators._compute.(UNI_DIST, Vector(d_input))

    GPUEventGenerators._compute!(UNI_DIST, d_vals, d_input)

    @test isapprox(Vector(d_vals), h_groundtruth)
end

struct TestMultivariateTarget{S} <: GPUEventGenerators.AbstractMultivariateTarget{S} end
GPUEventGenerators._compute(::TestMultivariateTarget, x) = cos(prod(x))


function testsuite_multivariate_target(backend, vec_type, el_type, N)
    DIMS = (1, rand(RNG, 2:5))

    @testset "dim = $dim" for dim in DIMS
        MULT_DIST = TestMultivariateTarget{dim}()
        d_input = vec_type(rand(RNG, NTuple{dim,el_type}, N))
        d_vals = vec_type(rand(RNG, el_type, N))

        h_groundtruth = GPUEventGenerators._compute.(MULT_DIST, Vector(d_input))

        GPUEventGenerators._compute!(MULT_DIST, d_vals, d_input)

        @test isapprox(Vector(d_vals), h_groundtruth)
    end
end
