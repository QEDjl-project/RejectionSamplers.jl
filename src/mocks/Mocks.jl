module Mocks

export MockProposal
export MockTarget

using Distributions
using Random
using StaticArrays
using KernelAbstractions
using RejectionSamplers
using Adapt

include("utils.jl")
include("sampler.jl")
include("proposal.jl")
include("target.jl")
include("buffers.jl")

end
