PARAMETER_RNG = Xoshiro(137137)

function testsuite_abstract_sampler(backend, vec_type, val_type, w_type, batch_size)

    @testset "properties" begin
        test_weight = rand(PARAMETER_RNG, w_type)
        test_sampler = Mocks.MockSampler(val_type, test_weight)

        @test value_type(test_sampler) == val_type
        @test weight_type(test_sampler) == w_type
    end
    @testset "host side rng" begin
        @testset "rng: $(typeof(rng))" for rng in TestUtils.get_host_rngs(backend)

            @testset "buffer allocation" begin
                test_weight = rand(PARAMETER_RNG, w_type)
                test_sampler = Mocks.MockSampler(val_type, test_weight)

                test_buffer = allocate_buffer(rng, test_sampler, backend, batch_size)

                @test test_buffer isa SampleBuffer

                @test get_backend(test_buffer) isa typeof(backend)
                @test length(test_buffer) == batch_size
                @test eltype(test_buffer) == Sample{val_type, w_type}
                @test value_type(test_buffer) == val_type
                @test weight_type(test_buffer) == w_type

                @test value_type(test_buffer) == value_type(test_sampler)
                @test weight_type(test_buffer) == weight_type(test_sampler)
            end

            @testset "sampling" begin
                test_weight = rand(PARAMETER_RNG, w_type)
                test_sampler = Mocks.MockSampler(val_type, test_weight)

                groundtruth = Mocks.groundtruth_samples(val_type, test_sampler, batch_size)
                test_buffer = allocate_buffer(rng, test_sampler, backend, batch_size)

                rand!(rng, test_sampler, backend, test_buffer)

                # TODO: use getvalues and getweights after they are tested
                @test _isapprox(Array(test_buffer.samples.value), groundtruth.samples.value)
                @test _isapprox(Array(test_buffer.samples.weight), groundtruth.samples.weight)
            end
        end
    end
    @testset "device side rng" begin
        @testset "rng: $(typeof(rng))" for rng in TestUtils.get_device_rngs(backend)

            @testset "buffer allocation" begin
                test_weight = rand(PARAMETER_RNG, w_type)
                test_sampler = Mocks.MockSampler(val_type, test_weight)

                test_buffer = allocate_buffer(rng, test_sampler, backend, batch_size)

                @test test_buffer isa SampleBuffer

                @test get_backend(test_buffer) isa typeof(backend)
                @test length(test_buffer) == batch_size
                @test eltype(test_buffer) == Sample{val_type, w_type}
                @test value_type(test_buffer) == val_type
                @test weight_type(test_buffer) == w_type

                @test value_type(test_buffer) == value_type(test_sampler)
                @test weight_type(test_buffer) == weight_type(test_sampler)
            end

            @testset "batch sample" begin
                test_weight = rand(PARAMETER_RNG, w_type)
                test_sampler = Mocks.MockSampler(val_type, test_weight)

                groundtruth = Mocks.groundtruth_samples(val_type, test_sampler, batch_size)
                test_buffer = allocate_buffer(rng, test_sampler, backend, batch_size)

                rand!(rng, test_sampler, backend, test_buffer)

                # TODO: use getvalues and getweights after they are tested
                @test _isapprox(Array(test_buffer.samples.value), groundtruth.samples.value)
                @test _isapprox(Array(test_buffer.samples.weight), groundtruth.samples.weight)
            end
            @testset "single sample" begin

            end
        end
    end
    return nothing
end

function testsuite_transformbased_sampler(backend, vec_type, val_type, w_type, batch_size)
    @testset "properties" begin
        test_weight = rand(PARAMETER_RNG, w_type)
        test_sampler = Mocks.MockTransformBasedSampler(val_type, test_weight)

        @test value_type(test_sampler) == val_type
        @test weight_type(test_sampler) == w_type
    end
    @testset "host side rng" begin
        @testset "rng: $(typeof(rng))" for rng in TestUtils.get_host_rngs(backend)

            @testset "buffer allocation" begin
                test_weight = rand(PARAMETER_RNG, w_type)
                test_sampler = Mocks.MockTransformBasedSampler(val_type, test_weight)

                test_buffer = allocate_buffer(rng, test_sampler, backend, batch_size)

                @test test_buffer isa SampleBuffer

                @test get_backend(test_buffer) isa typeof(backend)
                @test length(test_buffer) == batch_size
                @test eltype(test_buffer) == Sample{val_type, w_type}
                @test value_type(test_buffer) == val_type
                @test weight_type(test_buffer) == w_type

                @test value_type(test_buffer) == value_type(test_sampler)
                @test weight_type(test_buffer) == weight_type(test_sampler)
            end

            @testset "sampling" begin
                test_weight = rand(PARAMETER_RNG, w_type)
                test_sampler = Mocks.MockTransformBasedSampler(val_type, test_weight)

                groundtruth = Mocks.groundtruth_transformbased_samples(val_type, test_sampler, batch_size)
                test_buffer = allocate_buffer(rng, test_sampler, backend, batch_size)

                rand!(rng, test_sampler, backend, test_buffer)

                # TODO: use getvalues and getweights after they are tested
                @test _isapprox(Array(test_buffer.samples.value), groundtruth.samples.value)
                @test _isapprox(Array(test_buffer.samples.weight), groundtruth.samples.weight)
            end
        end
    end
    @testset "device side rng" begin
        @testset "rng: $(typeof(rng))" for rng in TestUtils.get_device_rngs(backend)

            @testset "buffer allocation" begin
                test_weight = rand(PARAMETER_RNG, w_type)
                test_sampler = Mocks.MockTransformBasedSampler(val_type, test_weight)

                test_buffer = allocate_buffer(rng, test_sampler, backend, batch_size)

                @test test_buffer isa SampleBuffer

                @test get_backend(test_buffer) isa typeof(backend)
                @test length(test_buffer) == batch_size
                @test eltype(test_buffer) == Sample{val_type, w_type}
                @test value_type(test_buffer) == val_type
                @test weight_type(test_buffer) == w_type

                @test value_type(test_buffer) == value_type(test_sampler)
                @test weight_type(test_buffer) == weight_type(test_sampler)
            end

            @testset "batch sample" begin
                test_weight = rand(PARAMETER_RNG, w_type)
                test_sampler = Mocks.MockTransformBasedSampler(val_type, test_weight)

                groundtruth = Mocks.groundtruth_transformbased_samples(val_type, test_sampler, batch_size)
                test_buffer = allocate_buffer(rng, test_sampler, backend, batch_size)

                rand!(rng, test_sampler, backend, test_buffer)

                # TODO: use getvalues and getweights after they are tested
                @test _isapprox(Array(test_buffer.samples.value), groundtruth.samples.value)
                @test _isapprox(Array(test_buffer.samples.weight), groundtruth.samples.weight)
            end
            @testset "single sample" begin

            end
        end
    end
    return nothing
end
