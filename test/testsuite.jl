include("filter_scan.jl")
include("proposal.jl")
include("target.jl")
include("generate.jl")
include("max_finder.jl")

function testsuite_run(backend, vec_type, el_type)
    @testset "filter scan" testsuite_filter_scan(
        backend,
        vec_type,
        el_type,
        NTuple{4, el_type},
    )
    @testset "proposal generation" begin
        testsuite_uniform_proposal(backend, vec_type, el_type, 256)
    end

    @testset "target evaluation" begin
        @testset "univariate target" testsuite_univariate_target(
            backend,
            vec_type,
            el_type,
            256,
        )
        @testset "multivariate target" testsuite_multivariate_target(
            backend,
            vec_type,
            el_type,
            256,
        )
    end

    @testset "Compton target" testsuite_Compton_target(backend, vec_type, el_type, 256)
    @testset "max finder" testsuite_max_finder_gaussian(backend, vec_type, el_type, 4)

    @testset "event generation" begin
        @testset "univariate generation" testsuite_univariate_generation(
            backend,
            vec_type,
            el_type,
            Int(2^12),
            Int(2^10),
        )
        @testset "multivariate generation" testsuite_multivariate_generation(
            backend,
            vec_type,
            el_type,
            Int(2^12),
            Int(2^10),
        )
        @testset "buffer allocation" begin
            # TODO: consider testing buffer for PSPs
            @testset "Scalar" testsuite_buffer_allocation(backend, el_type, el_type, 1024, 256)
            @testset "SVector" testsuite_buffer_allocation(backend, SVector{3, el_type}, el_type, 1024, 256)
            @testset "NTuple" testsuite_buffer_allocation(backend, NTuple{3, el_type}, el_type, 1024, 256)
        end
    end
    return nothing
end
