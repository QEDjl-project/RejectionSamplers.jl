abstract type AbstractUnitCoordsProposal{T,N} <: AbstractProposal{T,CoordinateVariate} end

function degree_of_freedom(::AbstractUnitCoordsProposal{T,N}) where {T,N}
    return N
end

"""

    UnitCoordsProposal{T,N}()

Proposal generator, which generates unit coordinate vectors with element type `T` and length `N`.
"""
struct UnitCoordsProposal{T,N} <: AbstractUnitCoordsProposal{T,N} end
UnitCoordsProposal(proposal::AbstractUnitCoordsProposal) = proposal
UnitCoordsProposal(proposal::AbstractProposal) =
    UnitCoordsProposal{eltype(proposal),degree_of_freedom(proposal)}()

# cpu version
function _rand!(
    rng::AbstractRNG,
    proposal::UnitCoordsProposal{T,N},
    dest::AbstractVector{TT},
) where {T,N,TT<:SVector{N,T}}
    isempty(dest) && return dest

    for i in eachindex(dest)
        dest[i] = sacollect(TT, _ -> rand(rng))
    end
    return A
end

# gpu version
function gpu_rand(
    proposal::UnitCoordsProposal{T,N},
    threadid,
    randstate::AbstractVector{NTuple{4,UInt32}},
) where {T,N}
    return sacollect(SVector{N,T}, _ -> GPUArrays.gpu_rand(T, threadid, randstate))
end

function _rand!(
    rng::GPUArrays.RNG,
    proposal::UnitCoordsProposal{T,N},
    A::GPUArrays.AnyGPUArray{TT},
) where {T,N,TT<:SVector{N,T}}
    isempty(A) && return A
    @kernel function rand!(a, randstate)
        idx = @index(Global, Linear)
        @inbounds a[idx] = gpu_rand(T, ((idx - 1) % length(randstate) + 1), randstate)
    end
    rand!(get_backend(A))(A, rng.state; ndrange = size(A))
    return A
end
