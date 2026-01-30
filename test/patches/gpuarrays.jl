function testsuite_gpuarrays_rand(backend, vec_type, val_type, size)

    # these tests are GPU only
    if backend isa CPU
        return nothing
    end

    payload = allocate(backend, val_type, size)
    payload_untouched = deepcopy(payload)

    test_rng = GPUArrays.default_rng(vec_type)

    rand!(test_rng, payload)

    @test !all(_isapprox.(Array(payload_untouched), Array(payload)))

    return nothing
end
