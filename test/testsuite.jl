
include("filter_scan.jl")
include("proposal.jl")
include("target.jl")
include("generate.jl")

function testsuite_run(backend, vec_type, el_type)
    @testset "filter scan" testsuite_filter_scan(
        backend,
        vec_type,
        el_type,
        NTuple{4,el_type},
    )
    @testset "proposal" testsuite_proposal(backend, vec_type, el_type, 256)
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
    @testset "Compton target" testsuite_Compton_target(backend, vec_type, el_type, 256)
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
end
