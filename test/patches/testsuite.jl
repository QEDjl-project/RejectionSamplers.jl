include("gpuarrays.jl")

if VERSION >= v"1.11"
    _get_value_types(el_type, ::Val{N} = Val(4)) where {N} = (el_type, SVector{N, el_type}, NTuple{N, el_type})
else
    # Julia 1.10 does not support random generation of NTuple on CPU
    @warn "Julia 1.10 does not support random generation of NTuple on CPU"
    _get_value_types(el_type, ::Val{N} = Val(4)) where {N} = (el_type, SVector{N, el_type})
end

function testsuite_patches_run(backend, vec_type, el_type)

    return @testset "val_type: $val_type" for val_type in _get_value_types(el_type)

        @testset "GPUArrays.RNG rand!" testsuite_gpuarrays_rand(backend, vec_type, val_type, 256)

    end
end
