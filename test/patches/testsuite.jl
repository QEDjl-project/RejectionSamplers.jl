include("gpuarrays.jl")

function testsuite_patches_run(backend, vec_type, el_type)

    return @testset "val_type: $val_type" for val_type in _get_value_types(el_type)

        @testset "GPUArrays.RNG rand!" testsuite_gpuarrays_rand(backend, vec_type, val_type, 256)

    end
end
