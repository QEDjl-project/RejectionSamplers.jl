struct MockTarget <: GPUEventGenerators.AbstractTargetDistribution end


function _mock_target(x::Real)
    return x < 0 ? zero(x) : one(x)
end

# scalar
function GPUEventGenerators._compute!(
        ::MockTarget,
        dest::AbstractArray{T},
        x::AbstractArray{TT}
    ) where {
        T <: Real,
        TT <: Real,
    }

    return @inline broadcast!(_mock_target, dest, x)
end

# SVector
function _mock_target(x::SVector{N, T}) where {N, T <: Real}
    return mapreduce(_mock_target, +, x)
end

# NOTE: even if trivially the same as for scalars, this will stay to be optimized later
function GPUEventGenerators._compute!(
        ::MockTarget,
        dest::AbstractArray{T},
        x::AbstractArray{TT}
    ) where {
        D,
        T <: Real,
        TT <: SVector{D, T},
    }

    return @inline broadcast!(_mock_target, dest, x)
end

# NTuple
# NOTE: even if trivially the same as for SVector, this will stay to be optimized later
function _mock_target(x::NTuple{N, T}) where {N, T <: Real}
    return mapreduce(_mock_target, +, x)
end

function GPUEventGenerators._compute!(
        ::MockTarget,
        dest::AbstractArray{T},
        x::AbstractArray{TT}
    ) where {
        D,
        T <: Real,
        TT <: NTuple{D, T},
    }

    return @inline broadcast!(_mock_target, dest, x)
end
