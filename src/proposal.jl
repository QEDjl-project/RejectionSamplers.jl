

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

"""
abstract type AbstractProposal{T} <: Sampleable{Univariate,Continuous} end
Base.eltype(::AbstractProposal{T}) where {T} = T

# Univariate distributions

abstract type AbstractUnivariateProposal{T} <: AbstractProposal{T} end

# Uniform univariate proposal

struct UniformProposal{T<:Real,DIST} <: AbstractUnivariateProposal{T}
    dist::DIST

    function UniformProposal(a::T, b::T) where {T<:Real}
        dist = Uniform(a, b)
        return new{T,typeof(dist)}(dist)
    end
end
UniformProposal(d::NTuple{2}) = UniformProposal(d...)

Base.extrema(p::UniformProposal) = extrema(p.dist)
Base.minimum(p::UniformProposal) = minimum(p.dist)
Base.maximum(p::UniformProposal) = maximum(p.dist)

# CPU only?
Distributions.rand(rng::AbstractRNG, d::UniformProposal) = rand(rng, d.dist)

# Random Interface (important for GPU)
function Distributions._rand!(
    rng::AbstractRNG,
    d::UniformProposal{T},
    A::AbstractArray{T},
) where {T<:Real}

    # fallback on Distributions.Uniform (which works on GPU)
    rand!(rng, d.dist, A)
end
