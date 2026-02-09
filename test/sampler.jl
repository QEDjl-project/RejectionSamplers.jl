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
    #=
    @testset "sampling" begin
        @testset "single implementation" begin
            test_weight = rand(PARAMETER_RNG, w_type)
            test_sampler = Mocks.MockSamplerSingle(val_type, test_weight)

            if backend isa CPU
                @testset "single draw (host, CPU only)" begin
                    rng = Random.default_rng()
                    single_sample = rand_single(rng, test_sampler)
                    groundtruth_single = Mocks.groundtruth_sinlge_sample(val_type)

                    @test isapprox(single_sample.value, groundtruth_single)
                    @test typeof(single_sample.value) == typeof(groundtruth_single)
                end
            end

            @testset "batch draw" begin
                groundtruth = fill(Mocks.groundtruth_sinlge_sample(val_type), batch_size)
                sample_batch = allocate_samples(backend, val_type, w_type, batch_size)
                #rng = RejectionSamplers.default_rng(typeof(sample_batch.value))
                rand!(test_sampler, sample_batch)

                @test isapprox(Array(sample_batch.value), groundtruth)
            end
        end

        @testset "batch implementation" begin
            test_weight = rand(PARAMETER_RNG, w_type)
            test_sampler = Mocks.MockSampler(val_type, test_weight)

            groundtruth = Mocks.groundtruth_samples(val_type, batch_size)
            test_samples = allocate_samples(backend, val_type, w_type, batch_size)

            rand!(test_sampler, test_samples)

            @test isapprox(Array(test_samples.value), groundtruth)
        end
    end
=#
    return nothing
end
