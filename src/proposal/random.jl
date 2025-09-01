# generic fallback to Random.rand! (used on CPU)
_rand!(rng::AbstractRNG, a::Vector) = Random.rand!(rng, a)

function gpu_rand(
        ::Type{TT},
        threadid,
        randstate::AbstractVector{NTuple{4, UInt32}},
    ) where {N, T, TT <: SVector{N, T}}
    #return sacollect(TT, _ -> GPUArrays.gpu_rand(T, threadid, randstate))
    return TT(ntuple(x -> GPUArrays.gpu_rand(T, threadid, randstate), N))
end

function _rand!(
        rng::GPUArrays.RNG,
        A::GPUArrays.AnyGPUArray{TT},
    ) where {T, N, TT <: SVector{N, T}}
    isempty(A) && return A
    @kernel function rand!(a, randstate)
        idx = @index(Global, Linear)
        @inbounds a[idx] = gpu_rand(TT, ((idx - 1) % length(randstate) + 1), randstate)
    end
    rand!(get_backend(A))(A, rng.state; ndrange = size(A))
    return A
end

# TODO:
# - implement own instance of default_rng


"""

    default_rng(v::AbstractVector)

Return default rng to randomize given vector. Uses fallbacks on `Random.default_rng` and `GPUArrays.default_rng`.
"""
function default_rng end

default_rng(::Type{T}) where {T} = Random.default_rng()
default_rng(::Type{V}) where {V <: GPUArrays.AnyGPUArray} = GPUArrays.default_rng(V)
