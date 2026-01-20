function groundtruth_samples(::Type{T}, size) where {T <: Real}
    steps = T(2 / (size + 1))
    return StepRangeLen(-1 + steps, steps, size)
end

function groundtruth_samples(::Type{VALTYPE}, size) where {T, N, VALTYPE <: SVector{N, T}}
    steps = T(2 / (size + 1))
    return value = map(i -> fill(-1 + i * steps, VALTYPE), 1:size)
end

function groundtruth_samples(::Type{VALTYPE}, size) where {T, N, VALTYPE <: NTuple{N, T}}
    steps = T(2 / (size + 1))
    return value = map(i -> ntuple(x -> -1 + i * steps, N), 1:size)
end

groundtruth_sinlge_sample(::Type{VALTYPE}) where {VALTYPE <: Real} = -one(VALTYPE)
groundtruth_sinlge_sample(::Type{VALTYPE}) where {T, N, VALTYPE <: SVector{N, T}} = .- ones(VALTYPE)
groundtruth_sinlge_sample(::Type{VALTYPE}) where {T, N, VALTYPE <: NTuple{N, T}} = ntuple(x -> - one(T), N)
