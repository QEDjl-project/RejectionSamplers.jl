"""

    AbstractSample{Ts,Tw}

Abstract base type for samples holding values of type `Ts` and weight of type `Tw`.

It must implement the following interface functions:

* [`value`](@ref): return the value of the sample (must have type `Ts`)
* [`weight`](@ref): return the weight of the sample (must have type `Tw`)
"""
abstract type AbstractSample{Ts, Tw} end

"""
    weight(::AbstractSample)

Interface function for abstract samples. Return the weight of a given sample. Must return `one(dtype)` if the sample has no weight.
"""
function weight end

"""
    value(::AbstractSample)

Interface function for abstract samples. Returns the value of the sample.
"""
function value end
