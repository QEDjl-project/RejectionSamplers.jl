module GPUEventGenerators

# utils
export filter_scan

# proposal
export UniformUnivariateProposal
export UniformMultivariateProposal
export GlobalTransformUniformProposal, LocalTransformUniformProposal

# target
export maximum_value

# maximum finding
export NaiveMaxFinder, QuantileReductionMethod


using Distributions
using KernelAbstractions
using Atomix
using Random
using QEDbase
using QEDcore
using GPUArrays
using StaticArrays
using StaticArrays: sacollect

include("filter_scan.jl")

include("proposal/random.jl")
include("proposal/interface.jl")
include("proposal/generics.jl")
include("proposal/uniform.jl")

include("target.jl")
include("generate.jl")

# max finding
include("max_finder/types.jl")
include("max_finder/findmax.jl")
include("max_finder/naive.jl")
include("max_finder/quantile_reduction.jl")


include("plotting.jl")

include("examples/truncated_gaussian.jl")
include("examples/perturbative_compton.jl")

include("testutils/TestUtils.jl")

include("patches.jl")

end
