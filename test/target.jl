
RNG = Xoshiro(137137)
N = 256

struct TestTarget <: GPUEventGenerators.AbstractUnivariatTargetDistribution end
GPUEventGenerators._compute(::TestTarget, x) = cos(x)

DIST = TestTarget()

function testsuite_target(backend, vec_type, el_type, N)
    d_input = vec_type(rand(RNG, el_type, N))
    d_vals = similar(d_input)

    h_groundtruth = GPUEventGenerators._compute.(DIST, Vector(d_input))

    GPUEventGenerators._compute!(DIST, d_vals, d_input)

    @test isapprox(Vector(d_vals), h_groundtruth)
end
