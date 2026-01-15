PARAMETER_RNG = Xoshiro(137137)

function testsuite_abstract_sampler(backend, value_type, weight_type, batch_size)
    test_weight = rand(PARAMETER_RNG, weight_type)
    test_sampler = Mocks.MockSampler(value_type, test_weight)
    test_samples = allocate_samples(backend, value_type, weight_type, batch_size)
    groundtruth = Mocks.groundtruth_samples(value_type, batch_size)

    rand!(test_sampler, test_samples)

    return @test isapprox(Array(test_samples.value), groundtruth)

end
