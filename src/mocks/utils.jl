function groundtruth_samples(::Type{T}, sampler, size) where {T <: Real}
    steps = T(2 / (size + 1))
    return SampleBuffer(
        StepRangeLen(-1 + steps, steps, size),
        fill(sampler.weight, size)
    )
end

function groundtruth_samples(::Type{VALTYPE}, sampler, size) where {T, N, VALTYPE <: SVector{N, T}}
    steps = T(2 / (size + 1))
    return SampleBuffer(
        map(i -> fill(-1 + i * steps, VALTYPE), 1:size),
        fill(sampler.weight, size)
    )
end

function groundtruth_samples(::Type{VALTYPE}, sampler, size) where {T, N, VALTYPE <: NTuple{N, T}}
    steps = T(2 / (size + 1))
    return SampleBuffer(
        map(i -> ntuple(x -> -1 + i * steps, Val(N)), 1:size),
        fill(sampler.weight, size)
    )
end

groundtruth_sinlge_sample(::Type{VALTYPE}) where {VALTYPE <: Real} = -one(VALTYPE)
groundtruth_sinlge_sample(::Type{VALTYPE}) where {T, N, VALTYPE <: SVector{N, T}} = .- ones(VALTYPE)
groundtruth_sinlge_sample(::Type{VALTYPE}) where {T, N, VALTYPE <: NTuple{N, T}} = ntuple(x -> - one(T), Val(N))
