module Mocks

export MockProposal
export MockTarget

using Distributions
using Random
using StaticArrays
using GPUEventGenerators

include("proposal.jl")
include("target.jl")

end
