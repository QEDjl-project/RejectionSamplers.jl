
RNG = Xoshiro(628)

N = 256

# FIXME: avoid piracy
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

function testsuite_filter_scan(backend, vec_type, el_type, payload_type)

    payload = vec_type(rand(RNG, payload_type, N))
    weights = vec_type(rand(RNG, el_type, N) .* el_type(1.05)) # calculated "probability", make slightly larger to test the behaviour with weights > 1
    random_numbers = vec_type(rand(RNG, el_type, N)) # random values to filter against

    out_payload = vec_type(zeros(eltype(payload), size(payload)))   # output buffer 1
    out_weights = vec_type(zeros(eltype(weights), size(weights)))   # output buffer 2
    accepted_count = vec_type(zeros(Int32, 1)) # global memory counter

    # TODO: is 32 the best number? (KA uses 128)

    # call kernel
    filter_scan(backend, 32)(
        payload,
        weights,
        random_numbers,
        out_payload,
        out_weights,
        accepted_count;
        ndrange = N,
    )
    KernelAbstractions.synchronize(backend)

    (expected_accepts, out_payload_gt, out_weights_gt) =
        _groundtruth_filter_scan(Vector(payload), Vector(weights), Vector(random_numbers))

    @test Vector(accepted_count)[1] == expected_accepts
    @test all(sort(Vector(out_payload)) .== sort(out_payload_gt))
    @test all(sort(Vector(out_weights)) .== sort(out_weights_gt))

    # second call, only make buffers larger but otherwise keep going

    payload = vec_type(rand(RNG, payload_type, N))
    weights = vec_type(rand(RNG, el_type, N) .* el_type(1.05))
    random_numbers = vec_type(rand(RNG, el_type, N))

    append!(out_payload, zeros(eltype(payload), expected_accepts))
    append!(out_weights, zeros(eltype(weights), expected_accepts))

    # call kernel
    filter_scan(backend, 32)(
        payload,
        weights,
        random_numbers,
        out_payload,
        out_weights,
        accepted_count;
        ndrange = N,
    )
    KernelAbstractions.synchronize(backend)

    (expected_accepts_2, out_payload_gt, out_weights_gt) =
        _groundtruth_filter_scan(Vector(payload), Vector(weights), Vector(random_numbers))

    @test Vector(accepted_count)[1] == expected_accepts + expected_accepts_2
    @test all(sort(Vector(out_payload[(expected_accepts+1):end])) .== sort(out_payload_gt))
    @test all(sort(Vector(out_weights[(expected_accepts+1):end])) .== sort(out_weights_gt))
end
