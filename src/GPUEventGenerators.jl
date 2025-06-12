module GPUEventGenerators

# utils
export filter_scan

# proposal
export UniformUnivariateProposal
export UniformMultivariateProposal

# target
export maximum_value


using Distributions
using KernelAbstractions
using Atomix
using Random
using GPUArrays

include("filter_scan.jl")
include("proposal.jl")
include("target.jl")
include("generate.jl")

include("examples/truncated_gaussian.jl")
include("examples/perturbative_compton.jl")

include("testutils/TestUtils.jl")

end
