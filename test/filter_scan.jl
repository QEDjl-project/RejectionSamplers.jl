using GPUEventGenerators
using Random
using KernelAbstractions

RNG = MersenneTwister(628)

N = 256

function _groundtruth_filter_scan(in_Q, in_T, in_RNG)
    c = 0

    out_Q = zeros(eltype(in_Q), size(in_Q))   # output buffer 1
    out_T = zeros(eltype(in_T), size(in_T))   # output buffer 2
    for i in eachindex(in_Q)
        if in_T[i] < in_RNG[i]
            c += 1
            out_Q[c] = in_Q[i]
            out_T[c] = in_T[i]
        end
    end
    return (c, out_Q, out_T)
end

@testset "testing with $VECTOR_T" for VECTOR_T in VECTOR_TYPES
    @info "Starting $VECTOR_T tests..."
    @testset "filtering scan with $T" for T in FLOAT_TYPES[VECTOR_T]
        # we don't really care for the test what the payload is
        in_Q = VECTOR_T(rand(RNG, T, N)) # "payload"
        in_T = VECTOR_T(rand(RNG, T, N)) # calculated "probability"
        in_RNG = VECTOR_T(rand(RNG, T, N)) # random values to filter against

        out_Q = VECTOR_T(zeros(eltype(in_Q), size(in_Q)))   # output buffer 1
        out_T = VECTOR_T(zeros(eltype(in_T), size(in_T)))   # output buffer 2
        g_out_size = VECTOR_T(zeros(Int64, 1)) # global memory counter

        BACKEND = get_backend(in_Q)

        # call kernel
        filter_scan(BACKEND, 32)(in_Q, in_T, in_RNG, out_Q, out_T, g_out_size; ndrange = N)
        KernelAbstractions.synchronize(BACKEND)

        (expected_accepts, out_Q_gt, out_T_gt) =
            _groundtruth_filter_scan(Vector(in_Q), Vector(in_T), Vector(in_RNG))

        @test Vector(g_out_size)[1] == expected_accepts
        @test all(sort(Vector(out_Q)) .== sort(out_Q_gt))
        @test all(sort(Vector(out_T)) .== sort(out_T_gt))

        # second call, only make buffers larger but otherwise keep going

        in_Q = VECTOR_T(rand(RNG, T, N))
        in_T = VECTOR_T(rand(RNG, T, N))
        in_RNG = VECTOR_T(rand(RNG, T, N))

        append!(out_Q, zeros(eltype(in_Q), expected_accepts))
        append!(out_T, zeros(eltype(in_T), expected_accepts))

        # call kernel
        filter_scan(BACKEND, 32)(in_Q, in_T, in_RNG, out_Q, out_T, g_out_size; ndrange = N)
        KernelAbstractions.synchronize(BACKEND)

        (expected_accepts_2, out_Q_gt, out_T_gt) =
            _groundtruth_filter_scan(Vector(in_Q), Vector(in_T), Vector(in_RNG))

        @test Vector(g_out_size)[1] == expected_accepts + expected_accepts_2
        @test all(sort(Vector(out_Q[(expected_accepts+1):end])) .== sort(out_Q_gt))
        @test all(sort(Vector(out_T[(expected_accepts+1):end])) .== sort(out_T_gt))
    end
end
