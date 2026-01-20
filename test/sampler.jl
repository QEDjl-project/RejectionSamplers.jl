PARAMETER_RNG = Xoshiro(137137)

function testsuite_abstract_sampler(backend, val_type, w_type, batch_size)

    @testset "properties" begin
        test_weight = rand(PARAMETER_RNG, w_type)
        test_sampler = Mocks.MockSampler(val_type, test_weight)

        @test value_type(test_sampler) == val_type
        @test weight_type(test_sampler) == w_type
    end

    @testset "sample allocation" begin
        test_weight = rand(PARAMETER_RNG, w_type)
        test_sampler = Mocks.MockSampler(val_type, test_weight)

        test_samples = allocate_samples(backend, test_sampler, batch_size)
        groundtruth = allocate_samples(backend, val_type, w_type, batch_size)

        @test test_samples isa SampleVector
        @test typeof(test_samples) == typeof(groundtruth)

        @test get_backend(test_samples.value) == get_backend(groundtruth.value)
        @test get_backend(test_samples.weight) == get_backend(groundtruth.weight)

        @test size(test_samples.value) == size(groundtruth.value)
        @test size(test_samples.weight) == size(groundtruth.weight)

        @test eltype(test_samples.value) == eltype(groundtruth.value)
        @test eltype(test_samples.weight) == eltype(groundtruth.weight)
    end

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

    return nothing
end
