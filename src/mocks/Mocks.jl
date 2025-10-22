module Mocks

export MockProposal
export MockTarget

using Distributions
using Random
using StaticArrays
using KernelAbstractions
using RejectionSamplers

include("proposal.jl")
include("target.jl")

end
