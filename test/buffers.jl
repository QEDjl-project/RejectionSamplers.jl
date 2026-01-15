CPU_RNG = Xoshiro(161)

@kernel function _test_setindex(buf, origin)
    idx = @index(Global)
    @inbounds buf[idx] = origin[idx]
end

@kernel function _test_setvalues(buf, origin)
    idx = @index(Global)
    @inbounds setvalue!(buf, origin[idx], idx)
end

@kernel function _test_setweights(buf, origin)
    idx = @index(Global)
    @inbounds setweight!(buf, origin[idx], idx)
end

@kernel function _test_getindex(buf, dest)
    idx = @index(Global)

    @inbounds dest[idx] = buf[idx]
end

@kernel function _test_getvalues(buf, dest)
    idx = @index(Global)

    @inbounds dest[idx] = getvalue(buf, idx)
end

@kernel function _test_getweights(buf, dest)
    idx = @index(Global)

    @inbounds dest[idx] = getweight(buf, idx)
end




function testsuite_buffer_interface(backend, val_type, size)

    @testset "properties" begin
        arr = allocate(backend, val_type, (size,))
        mock_buffer = Mocks.MockArrayBuffer(arr)

        @test get_backend(mock_buffer) isa typeof(backend)
        @test length(mock_buffer) == size
        @test eltype(mock_buffer) == val_type
    end

    @testset "getindex" begin

        # we generate randoms on CPU to be more stable across backends
        h_arr = allocate(CPU(), val_type, (size,))
        rand!(CPU_RNG, h_arr)

        d_arr = allocate(backend, val_type, (size,))
        copyto!(d_arr, h_arr)
        mock_buffer = Mocks.MockArrayBuffer(d_arr)

        dest = allocate(backend, val_type, (size,))
        _test_getindex(backend)(mock_buffer, dest; ndrange = length(mock_buffer))

        @test _isapprox(Array(dest), h_arr)
    end

    @testset "setindex" begin

        # we generate randoms on CPU to be more stable across backends
        h_input = allocate(CPU(), val_type, (size,))
        rand!(CPU_RNG, h_input)
        d_input = allocate(backend, val_type, (size,))
        copyto!(d_input, h_input)

        d_arr = allocate(backend, val_type, (size,))
        mock_buffer = Mocks.MockArrayBuffer(d_arr)

        _test_setindex(backend)(mock_buffer, d_input; ndrange = length(mock_buffer))

        @test _isapprox(Array(mock_buffer.data), h_input)
    end

    return nothing
end

function testsuite_sample_buffer_interface(backend, val_type, w_type, size)

    @testset "properties" begin
        values = allocate(backend, val_type, (size,))
        weights = allocate(backend, w_type, (size,))
        mock_buffer = Mocks.MockSampleBuffer(values, weights)

        @test get_backend(mock_buffer) isa typeof(backend)
        @test length(mock_buffer) == size
        @test eltype(mock_buffer) == Sample{val_type, w_type}
        @test value_type(mock_buffer) == val_type
        @test weight_type(mock_buffer) == w_type
    end

    @testset "getter" begin

        # we generate randoms on CPU to be more stable across backends
        h_values = allocate(CPU(), val_type, (size,))
        rand!(CPU_RNG, h_values)

        h_weights = allocate(CPU(), w_type, (size,))
        rand!(CPU_RNG, h_weights)

        d_values = allocate(backend, val_type, (size,))
        copyto!(d_values, h_values)

        d_weights = allocate(backend, w_type, (size,))
        copyto!(d_weights, h_weights)

        mock_buffer = Mocks.MockSampleBuffer(d_values, d_weights)

        dest_values = allocate(backend, val_type, (size,))
        _test_getvalues(backend)(mock_buffer, dest_values; ndrange = length(mock_buffer))
        @test _isapprox(Array(dest_values), h_values)

        dest_weights = allocate(backend, w_type, (size,))
        _test_getweights(backend)(mock_buffer, dest_weights; ndrange = length(mock_buffer))
        @test _isapprox(Array(dest_weights), h_weights)

    end

    @testset "setindex" begin

        # input
        # we generate randoms on CPU to be more stable across backends
        h_values = allocate(CPU(), val_type, (size,))
        rand!(CPU_RNG, h_values)

        h_weights = allocate(CPU(), w_type, (size,))
        rand!(CPU_RNG, h_weights)

        d_values = allocate(backend, val_type, (size,))
        copyto!(d_values, h_values)

        d_weights = allocate(backend, w_type, (size,))
        copyto!(d_weights, h_weights)

        values = allocate(backend, val_type, (size,))
        weights = allocate(backend, w_type, (size,))
        mock_buffer = Mocks.MockSampleBuffer(values, weights)

        _test_setvalues(backend)(mock_buffer, d_values; ndrange = length(mock_buffer))
        @test _isapprox(Array(mock_buffer.values), h_values)

        _test_setweights(backend)(mock_buffer, d_weights; ndrange = length(mock_buffer))
        @test _isapprox(Array(mock_buffer.weights), h_weights)
    end

    return nothing
end

function testsuite_sample_buffer(backend, val_type, w_type, size)

    @testset "properties" begin
        buf = Mocks.SampleBuffer(backend, val_type, w_type, size)

        @test get_backend(buf) isa typeof(backend)
        @test length(buf) == size
        @test eltype(buf) == Sample{val_type, w_type}
        @test value_type(buf) == val_type
        @test weight_type(buf) == w_type
    end

    @testset "getter" begin

        # we generate randoms on CPU to be more stable across backends
        h_values = allocate(CPU(), val_type, (size,))
        rand!(CPU_RNG, h_values)

        h_weights = allocate(CPU(), w_type, (size,))
        rand!(CPU_RNG, h_weights)

        d_values = allocate(backend, val_type, (size,))
        copyto!(d_values, h_values)

        d_weights = allocate(backend, w_type, (size,))
        copyto!(d_weights, h_weights)

        buf = SampleBuffer(d_values, d_weights)

        dest_values = allocate(backend, val_type, (size,))
        _test_getvalues(backend)(buf, dest_values; ndrange = length(buf))
        @test _isapprox(Array(dest_values), h_values)

        dest_weights = allocate(backend, w_type, (size,))
        _test_getweights(backend)(buf, dest_weights; ndrange = length(buf))
        @test _isapprox(Array(dest_weights), h_weights)

    end

    @testset "setindex" begin

        # input
        # we generate randoms on CPU to be more stable across backends
        h_values = allocate(CPU(), val_type, (size,))
        rand!(CPU_RNG, h_values)

        h_weights = allocate(CPU(), w_type, (size,))
        rand!(CPU_RNG, h_weights)

        d_values = allocate(backend, val_type, (size,))
        copyto!(d_values, h_values)

        d_weights = allocate(backend, w_type, (size,))
        copyto!(d_weights, h_weights)

        values = allocate(backend, val_type, (size,))
        weights = allocate(backend, w_type, (size,))
        buf = SampleBuffer(values, weights)

        _test_setvalues(backend)(buf, d_values; ndrange = length(buf))
        @test _isapprox(Array(buf.samples.value), h_values)

        _test_setweights(backend)(buf, d_weights; ndrange = length(buf))
        @test _isapprox(Array(buf.samples.weight), h_weights)
    end

    return nothing
end
