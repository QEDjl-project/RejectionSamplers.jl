PARAMETER_RNG = Xoshiro(137137)

using StructArrays

_to_array(s::StructArray{Sample{Ts, Tw}}) where {Ts, Tw} = Array(Sample.(s.value, s.weight))

function testsuite_samples(backend, value_type, weight_type, batch_size)

    test_values = allocate(backend, value_type, (batch_size,))
    test_weights = allocate(backend, weight_type, (batch_size,))
    test_weights .= KernelAbstractions.ones(backend, weight_type, batch_size)

    test_samples_aos = Sample.(test_values, test_weights)

    test_samples_soa = allocate_samples(backend, value_type, weight_type, batch_size)
    test_samples_soa.value .= test_values
    test_samples_soa.weight .= test_weights

    @testset "properties" begin
        @test eltype(test_samples_soa) == Sample{value_type, weight_type}
        @test length(test_samples_soa) == batch_size
        @test eltype(test_samples_soa.value) == value_type
        @test length(test_samples_soa.value) == batch_size
        @test eltype(test_samples_soa.weight) == weight_type
        @test length(test_samples_soa.weight) == batch_size
    end

    return @testset "sanity check" begin
        test_samples_aos_h = Array(test_samples_aos)
        test_samples_soa_h = _to_array(test_samples_soa)
        @test all(test_samples_soa_h .== test_samples_aos_h)
    end

end
