RNG = Xoshiro(137137)

struct TestUnivariateTarget <: GPUEventGenerators.AbstractUnivariatTargetDistribution end
GPUEventGenerators._compute(::TestUnivariateTarget, x) = cos(x)

const UNI_DIST = TestUnivariateTarget()

function testsuite_univariate_target(backend, vec_type, el_type, N)
    d_input = vec_type(rand(RNG, el_type, N))
    d_vals = similar(d_input)

    h_groundtruth = GPUEventGenerators._compute.(UNI_DIST, Vector(d_input))

    GPUEventGenerators._compute!(UNI_DIST, d_vals, d_input)

    return @test isapprox(Vector(d_vals), h_groundtruth)
end

struct TestMultivariateTarget{S} <: GPUEventGenerators.AbstractMultivariateTarget{S} end
GPUEventGenerators._compute(::TestMultivariateTarget, x) = cos(prod(x))


function testsuite_multivariate_target(backend, vec_type, el_type, N)
    DIMS = (1, rand(RNG, 2:5))

    return @testset "dim = $dim" for dim in DIMS
        MULT_DIST = TestMultivariateTarget{dim}()
        d_input = vec_type(rand(RNG, SVector{dim, el_type}, N))
        d_vals = vec_type(rand(RNG, el_type, N))

        h_groundtruth = GPUEventGenerators._compute.(MULT_DIST, Vector(d_input))

        GPUEventGenerators._compute!(MULT_DIST, d_vals, d_input)

        @test isapprox(Vector(d_vals), h_groundtruth)
    end
end

function _rand_compton_coords(ELTYPE, RNG, om, N)
    return [
        SVector(om, rand(RNG, ELTYPE) * 2 - one(ELTYPE), rand(RNG, ELTYPE) * 2 * pi) for
            _ in 1:N
    ]
end

function testsuite_Compton_target(backend, vec_type, el_type, N)
    if el_type == Float16 && !(backend isa CPU)
        return nothing
    end
    OMS = (el_type(1.0e-3),)
    PROC = Compton()
    MODEL = PerturbativeQED()
    IN_PSL = ComptonRestSystem()
    OUT_PSL = ComptonSphericalLayout(IN_PSL)


    return @testset "om = $om" for om in OMS

        h_coords = _rand_compton_coords(el_type, RNG, om, N)
        d_coords = vec_type(h_coords)

        COMPTON_DIST = ComptonDistribution{el_type}(MODEL, OUT_PSL)

        d_vals = vec_type(rand(RNG, el_type, N))

        h_groundtruth = GPUEventGenerators._compute.(COMPTON_DIST, h_coords)

        GPUEventGenerators._compute!(COMPTON_DIST, d_vals, d_coords)

        @test isapprox(Vector(d_vals), h_groundtruth)
    end
end
