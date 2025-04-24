using GPUEventGenerators
using Random
using KernelAbstractions

RNG = MersenneTwister(628)

N = 256

@inline Base.zero(::Type{Tuple{}}) = ()
@inline Base.zero(::Type{Tuple{Vararg{T,N}}}) where {T,N} =
    (zero(T), zero(NTuple{N - 1,T})...)

function _groundtruth_filter_scan(payload, weights, random_numbers)
    c = 0

    out_payload = zeros(eltype(payload), size(payload))   # output buffer 1
    out_weights = zeros(eltype(weights), size(weights))   # output buffer 2
    for i in eachindex(payload)
        if random_numbers[i] < weights[i]
            c += 1
            out_payload[c] = payload[i]

            out_weights[c] = if weights[i] > one(eltype(weights))
                weights[i]
            else
                one(eltype(weights))
            end
        end
    end
    return (c, out_payload, out_weights)
end

@testset "testing with $VECTOR_T" for VECTOR_T in VECTOR_TYPES
    @info "Starting $VECTOR_T tests..."
    @testset "filtering scan with $T" for T in FLOAT_TYPES[VECTOR_T]
        PAYLOAD_T = NTuple{4,T}   # make a "bigger" payload type

        # we don't really care for the test what the payload is
        payload = VECTOR_T(rand(RNG, PAYLOAD_T, N))
        weights = VECTOR_T(rand(RNG, T, N) .* T(1.05)) # calculated "probability", make slightly larger to test the behaviour with weights > 1
        random_numbers = VECTOR_T(rand(RNG, T, N)) # random values to filter against

        out_payload = VECTOR_T(zeros(eltype(payload), size(payload)))   # output buffer 1
        out_weights = VECTOR_T(zeros(eltype(weights), size(weights)))   # output buffer 2
        accepted_count = VECTOR_T(zeros(Int32, 1)) # global memory counter

        BACKEND = get_backend(payload)

        # call kernel
        filter_scan(BACKEND, 32)(
            payload,
            weights,
            random_numbers,
            out_payload,
            out_weights,
            accepted_count;
            ndrange = N,
        )
        KernelAbstractions.synchronize(BACKEND)

        (expected_accepts, out_payload_gt, out_weights_gt) = _groundtruth_filter_scan(
            Vector(payload),
            Vector(weights),
            Vector(random_numbers),
        )

        @test Vector(accepted_count)[1] == expected_accepts
        @test all(sort(Vector(out_payload)) .== sort(out_payload_gt))
        @test all(sort(Vector(out_weights)) .== sort(out_weights_gt))

        # second call, only make buffers larger but otherwise keep going

        payload = VECTOR_T(rand(RNG, PAYLOAD_T, N))
        weights = VECTOR_T(rand(RNG, T, N) .* T(1.05))
        random_numbers = VECTOR_T(rand(RNG, T, N))

        append!(out_payload, zeros(eltype(payload), expected_accepts))
        append!(out_weights, zeros(eltype(weights), expected_accepts))

        # call kernel
        filter_scan(BACKEND, 32)(
            payload,
            weights,
            random_numbers,
            out_payload,
            out_weights,
            accepted_count;
            ndrange = N,
        )
        KernelAbstractions.synchronize(BACKEND)

        (expected_accepts_2, out_payload_gt, out_weights_gt) = _groundtruth_filter_scan(
            Vector(payload),
            Vector(weights),
            Vector(random_numbers),
        )

        @test Vector(accepted_count)[1] == expected_accepts + expected_accepts_2
        @test all(
            sort(Vector(out_payload[(expected_accepts+1):end])) .== sort(out_payload_gt),
        )
        @test all(
            sort(Vector(out_weights[(expected_accepts+1):end])) .== sort(out_weights_gt),
        )
    end
end
