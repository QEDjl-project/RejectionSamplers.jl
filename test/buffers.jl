CPU_RNG = Xoshiro(161)

@kernel function _test_setindex(buf, origin)
    idx = @index(Global)
    @inbounds buf[idx] = origin[idx]
end

@kernel function _test_getindex(buf, dest)
    idx = @index(Global)

    @inbounds dest[idx] = buf[idx]
end

function testsuite_buffer_interface(backend, value_type, size)

    @testset "properties" begin
        arr = allocate(backend, value_type, (size,))
        mock_buffer = Mocks.MockArrayBuffer(arr)

        @test get_backend(mock_buffer) isa typeof(backend)
        @test length(mock_buffer) == size
        @test eltype(mock_buffer) == value_type
    end

    @testset "getindex" begin

        # we generate randoms on CPU to be more stable across backends
        h_arr = allocate(CPU(), value_type, (size,))
        rand!(CPU_RNG, h_arr)

        d_arr = allocate(backend, value_type, (size,))
        copyto!(d_arr, h_arr)
        mock_buffer = Mocks.MockArrayBuffer(d_arr)

        dest = allocate(backend, value_type, (size,))
        _test_getindex(backend)(mock_buffer, dest; ndrange = length(mock_buffer))

        @test isapprox(Array(dest), h_arr)
    end

    return @testset "setindex" begin

        # we generate randoms on CPU to be more stable across backends
        h_input = allocate(CPU(), value_type, (size,))
        rand!(CPU_RNG, h_input)
        d_input = allocate(backend, value_type, (size,))
        copyto!(d_input, h_input)

        d_arr = allocate(backend, value_type, (size,))
        mock_buffer = Mocks.MockArrayBuffer(d_arr)

        _test_setindex(backend)(mock_buffer, d_input; ndrange = length(mock_buffer))

        @test isapprox(Array(mock_buffer.data), h_input)
    end
end
