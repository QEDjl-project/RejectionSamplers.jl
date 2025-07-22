"""

    AbstractProposal{T}

Proposal distribution super type with element type `T`.
Interface functions, which must be implemented

```
rand!(rng,dist,dest)
rand!(dist,dest) = rand!(Random.default_rng(),dist,dest)
```

which generates random numbers and stores them in `dest`. Possible `dest` types are

- `Vector`
- `CuVector`
- `ROCArray`
- `OneArray`
- `MtlArray`

Depending on `rng`, the random numbers are generated directly on device, or they are generated on
cpu and copyied to the device.

!!! note

    Even if this is a subtype of `Sampleable{Univariate}`, we distinguish **univariate**
    proposal, which have real element types `T<:Real`, and **multivariate** targets, if the
    element type is a `NTuple`. In the terminology of `Distributions.jl`, both are
    technically considered *univariate*. However, we differentiate them because our current data
    layout represents coordinates as a single `NTuple` and phase space points as collections of
    such `NTuples`.

"""
abstract type AbstractProposal{T} <: Sampleable{Univariate, Continuous} end
Base.broadcastable(d::AbstractProposal) = Ref(d)
Base.eltype(::AbstractProposal{T}) where {T} = T

# Univariate distributions

abstract type AbstractUnivariateProposal{T} <: AbstractProposal{T} end

# Uniform univariate proposal

struct UniformUnivariateProposal{T <: Real, DIST} <: AbstractUnivariateProposal{T}
    dist::DIST

    function UniformUnivariateProposal(a::T, b::T) where {T <: Real}
        dist = Uniform(a, b)
        return new{T, typeof(dist)}(dist)
    end
end
UniformUnivariateProposal(d::NTuple{2}) = UniformUnivariateProposal(d...)

Base.extrema(p::UniformUnivariateProposal) = extrema(p.dist)
Base.minimum(p::UniformUnivariateProposal) = minimum(p.dist)
Base.maximum(p::UniformUnivariateProposal) = maximum(p.dist)

# CPU only?
Distributions.rand(rng::AbstractRNG, d::UniformUnivariateProposal) = rand(rng, d.dist)

# Random Interface (important for GPU)
function Distributions._rand!(
        rng::AbstractRNG,
        d::UniformUnivariateProposal{T},
        A::AbstractArray{T},
    ) where {T <: Real}

    # fallback on Distributions.Uniform (which works on GPU)
    return rand!(rng, d.dist, A)
end

# Multivariate proposal

abstract type AbstractMultivariateProposal{T} <: AbstractProposal{T} end

# Uniform multivariate proposal

function _assert_correct_boundaries(::Tuple{}, ::Tuple{}) end

function _assert_correct_boundaries(
        low::Tuple{Vararg{T, N}},
        up::Tuple{Vararg{T, N}},
    ) where {T <: Real, N}
    first(low) <= first(up) || throw(
        ArgumentError(
            "lower boundary need to be smaller or equal to the respective upper boundary",
        ),
    )
    return _assert_correct_boundaries(low[2:end], up[2:end])
end

struct UniformMultivariateProposal{T, N} <: AbstractMultivariateProposal{T}
    low::NTuple{N, T}
    up::NTuple{N, T}
    function UniformMultivariateProposal(low::NTuple{N, T}, up::NTuple{N, T}) where {T, N}
        _assert_correct_boundaries(low, up)
        return new{T, N}(low, up)
    end
end

UniformMultivariateProposal(low::AbstractVector, high::AbstractVector) =
    UniformMultivariateProposal(Tuple(low), Tuple(high))

Base.extrema(p::UniformMultivariateProposal) = (minimum(p), maximum(p))
Base.minimum(p::UniformMultivariateProposal) = p.low
Base.maximum(p::UniformMultivariateProposal) = p.up
Base.eltype(::UniformMultivariateProposal{T, N}) where {T, N} = NTuple{N, T}


# based on: https://github.com/JuliaGPU/GPUArrays.jl/blob/55a943ea5c876f6c34cb355eea17fb8290f81497/src/host/random.jl#L56
# not exported, therefore, no piracy
function gpu_rand(
        ::Type{NTuple{N, T}},
        threadid,
        randstate::AbstractVector{NTuple{4, UInt32}},
    ) where {N, T}
    return ntuple(x -> GPUArrays.gpu_rand(T, threadid, randstate), N)
end

# generic transformation of x in (0,1) to (low,high)
_transform_uniform_val(x, low, high) = (high - low) * x + low

# version which transforms result of gpu_rand to hyper cube of dist
function gpu_rand(
        ::Type{NTuple{N, T}},
        u::UniformMultivariateProposal,
        threadid,
        randstate::AbstractVector{NTuple{4, UInt32}},
    ) where {N, T}
    return ntuple(
        x ->
        (getindex(u.up, x) - getindex(u.low, x)) *
            GPUArrays.gpu_rand(T, threadid, randstate) + getindex(u.low, x),
        N,
    )
end

# based on: https://github.com/JuliaGPU/GPUArrays.jl/blob/55a943ea5c876f6c34cb355eea17fb8290f81497/src/host/random.jl#L84
# not exported, therefore, no piracy
# TODO:
function _rand!(rng::GPUArrays.RNG, A::GPUArrays.AnyGPUArray{T}) where {T <: Tuple}
    isempty(A) && return A
    @kernel function rand!(a, randstate)
        idx = @index(Global, Linear)
        @inbounds a[idx] = gpu_rand(T, ((idx - 1) % length(randstate) + 1), randstate)
    end
    rand!(get_backend(A))(A, rng.state; ndrange = size(A))
    return A
end

# version of _rand!, which propagates the dist to gpu_rand
# - figure out, if this should be from GPUArrays to allow correct dispatch for
# CUDA/AMDGPU/oneAPI
function _rand!(
        rng::GPUArrays.RNG,
        d::UniformMultivariateProposal{N, T},
        A::GPUArrays.AnyGPUArray{TT},
    ) where {N, T, TT <: Tuple}
    isempty(A) && return A
    @kernel function rand!(a, randstate)
        idx = @index(Global, Linear)
        @inbounds a[idx] = gpu_rand(TT, d, ((idx - 1) % length(randstate) + 1), randstate)
    end
    rand!(get_backend(A))(A, rng.state; ndrange = size(A))
    return A
end

function _rand!(
        rng::AbstractRNG,
        d::UniformMultivariateProposal{T, N},
        A::AbstractVector{TT},
    ) where {N, T, TT <: Tuple}
    isempty(A) && return A

    for i in eachindex(A)
        A[i] = _transform_uniform_val.(ntuple(_ -> rand(rng), N), d.low, d.up)
    end
    return A
end

# need to implement this to get the correct default_rng for Metal
function Random.rand!(
        d::UniformMultivariateProposal,
        A::GPUArrays.AnyGPUArray{TT},
    ) where {TT <: Tuple}
    rng = GPUArrays.default_rng(typeof(A))
    return _rand!(rng, d, A)
end

function Distributions._rand!(
        rng::AbstractRNG,
        d::UniformMultivariateProposal{T, N},
        A::AbstractArray{TT},
    ) where {N, T <: Real, TT <: Tuple}

    # fallback on Distributions.Uniform (which works on GPU)
    return _rand!(rng, d, A)
end

## TODO:
# - check if this works for CUDA or AMDGPU
# - consider using KA also for the CPU backend
