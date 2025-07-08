struct QuantileReductionMethod{T <: Real} <: AbstractSampleBasedMaxFinder
    p::T
    nsamples::Int
end

_nsamples(n::QuantileReductionMethod) = n.nsamples

function _findmax(
        method::QuantileReductionMethod{T},
        weights::AbstractVector{T},
    ) where {T <: Real}
    sorted_weights = sort(weights)
    sum_weight = sum(sorted_weights)
    sum_weight_quantile = method.p * sum_weight

<<<<<<< HEAD
    s = 0.0
    for weight in reverse(sorted_weights)
        s += weight
        if s >= sum_weight_quantile
            return weight
        end
    end

    # we should never reach this point
    return zero(T)
=======
    s = sum_weight
    for weight in sorted_weights
        s -= weight
        if s <= sum_weight_quantile
            return weight
        end
    end
    return
>>>>>>> f029ef4 (added max-finder)
end
