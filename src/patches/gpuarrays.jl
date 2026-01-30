## rand! implementation for NTuple and SVector for GPUArrays.RNG
# can be removed if added to GPUArrays or
# https://github.com/JuliaGPU/AMDGPU.jl/issues/881
# is solved.

function gpu_rand(
        ::Type{TT},
        threadid,
        randstate::AbstractVector{NTuple{4, UInt32}},
    ) where {N, T, TT <: SVector{N, T}}
    #return sacollect(TT, _ -> GPUArrays.gpu_rand(T, threadid, randstate))
    return TT(ntuple(x -> GPUArrays.gpu_rand(T, threadid, randstate), N))
end

function gpu_rand(
        ::Type{TT},
        threadid,
        randstate::AbstractVector{NTuple{4, UInt32}},
    ) where {N, T, TT <: NTuple{N, T}}
    return ntuple(x -> GPUArrays.gpu_rand(T, threadid, randstate), N)
end

function Random.rand!(
        rng::GPUArrays.RNG,
        A::GPUArrays.AnyGPUArray{TT},
    ) where {T, N, TT <: Union{SVector{N, T}, NTuple{N, T}}}
    isempty(A) && return A
    @kernel function rand!(a, randstate)
        idx = @index(Global, Linear)
        @inbounds a[idx] = gpu_rand(TT, ((idx - 1) % length(randstate) + 1), randstate)
    end
    rand!(get_backend(A))(A, rng.state; ndrange = size(A))
    return A
end
