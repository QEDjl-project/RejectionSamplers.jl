using GPUEventGenerators
using Random
using KernelAbstractions
using oneAPI

BACKEND = CPU()

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

@testset "filtering scan with $T" for T in [Float16, Float32, Float64]
    # we don't really care for the test what the payload is
    in_Q = rand(RNG, T, N) # "payload"
    in_T = rand(RNG, T, N) # calculated "probability"
    in_RNG = rand(RNG, T, N) # random values to filter against

    out_Q = zeros(eltype(in_Q), size(in_Q))   # output buffer 1
    out_T = zeros(eltype(in_T), size(in_T))   # output buffer 2
    g_out_size = zeros(Int64, 1) # global memory counter

    # call kernel
    filter_scan(BACKEND, 32)(in_Q, in_T, in_RNG, out_Q, out_T, g_out_size; ndrange = N)
    KernelAbstractions.synchronize(BACKEND)

    (expected_accepts, out_Q_gt, out_T_gt) = _groundtruth_filter_scan(in_Q, in_T, in_RNG)

    @test g_out_size[1] == expected_accepts
    @test out_Q == out_Q_gt
    @test out_T == out_T_gt

    # second call, only make buffers larger but otherwise keep going

    in_Q = rand(RNG, T, N)
    in_T = rand(RNG, T, N)
    in_RNG = rand(RNG, T, N)

    append!(out_Q, zeros(eltype(in_Q), N))
    append!(out_T, zeros(eltype(in_T), N))

    # call kernel
    filter_scan(BACKEND, 32)(in_Q, in_T, in_RNG, out_Q, out_T, g_out_size; ndrange = N)
    KernelAbstractions.synchronize(BACKEND)

    (expected_accepts_2, out_Q_gt, out_T_gt) = _groundtruth_filter_scan(in_Q, in_T, in_RNG)

    @test g_out_size[1] == expected_accepts + expected_accepts_2
    @test all([out_Q[x+expected_accepts] == out_Q_gt[x] for x = 1:N])
    @test all([out_T[x+expected_accepts] == out_T_gt[x] for x = 1:N])
end
