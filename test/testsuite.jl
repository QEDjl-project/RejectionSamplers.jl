include("testutils.jl")
include("samples.jl")
include("buffers.jl")
include("filter_scan.jl")
include("proposal.jl")
include("sampler.jl")
include("target.jl")
include("generate.jl")
include("max_finder.jl")

if VERSION >= v"1.11"
    _get_value_types(el_type, ::Val{N} = Val(4)) where {N} = (el_type, SVector{N, el_type}, NTuple{N, el_type})
else
    # Julia 1.10 does not support random generation of NTuple on CPU
    @warn "Julia 1.10 does not support random generation of NTuple on CPU"
    _get_value_types(el_type, ::Val{N} = Val(4)) where {N} = (el_type, SVector{N, el_type})
end

function testsuite_run(backend, vec_type, el_type)
    # TODO: every testsuite using val_type into this block
    @testset "val_type: $val_type" for val_type in _get_value_types(el_type)
        @testset "samples" testsuite_samples(
            backend,
            val_type,
            el_type,
            256
        )

        @testset "buffer interface" testsuite_buffer_interface(
            backend,
            val_type,
            256
        )

        @testset "sample buffer interface" testsuite_sample_buffer_interface(
            backend,
            val_type,
            el_type,
            256
        )

        @testset "sample buffer interface" testsuite_sample_buffer_interface(
            backend,
            val_type,
            el_type,
            256
        )

        @testset "sample buffer" testsuite_sample_buffer(
            backend,
            val_type,
            el_type,
            256
        )

        @testset "sampler" testsuite_abstract_sampler(
            backend,
            val_type,
            el_type,
            256
        )
    end

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

    @testset "max finder" testsuite_max_finder_gaussian(backend, vec_type, el_type, 4)

    @testset "event generation" begin
        @testset "sampler type" begin
            @testset "Scalar" testsuite_sampler(backend, el_type, el_type)
            @testset "Scalar" testsuite_sampler(backend, SVector{3, el_type}, el_type)
            @testset "Scalar" testsuite_sampler(backend, NTuple{3, el_type}, el_type)
        end
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
            @testset "Scalar" testsuite_buffer_allocation(backend, el_type, el_type, 256, 1024)
            @testset "SVector" testsuite_buffer_allocation(backend, SVector{3, el_type}, el_type, 256, 1024)
            @testset "NTuple" testsuite_buffer_allocation(backend, NTuple{3, el_type}, el_type, 256, 1024)
        end

        @testset "proposal generation" begin
            @testset "Scalar" testsuite_proposal_stage(backend, vec_type, el_type, el_type, 256)
            @testset "SVector" testsuite_proposal_stage(backend, vec_type, SVector{3, el_type}, el_type, 256)
            @testset "NTuple" testsuite_proposal_stage(backend, vec_type, NTuple{3, el_type}, el_type, 256)
        end

        @testset "target evaluation" begin
            @testset "Scalar" testsuite_target_stage(backend, vec_type, el_type, el_type, 256)
            @testset "SVector" testsuite_target_stage(backend, vec_type, SVector{3, el_type}, el_type, 256)
            @testset "NTuple" testsuite_target_stage(backend, vec_type, NTuple{3, el_type}, el_type, 256)
        end

        @testset "filter scan" begin
            @testset "Scalar" testsuite_filterscan_stage(backend, vec_type, el_type, el_type, 256, 1024)
            @testset "SVector" testsuite_filterscan_stage(backend, vec_type, SVector{3, el_type}, el_type, 256, 1024)
            @testset "NTuple" testsuite_filterscan_stage(backend, vec_type, NTuple{3, el_type}, el_type, 256, 1024)
        end
    end
    return nothing
end
