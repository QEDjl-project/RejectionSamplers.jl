module RejectionSamplers

# samples
export Sample, SampleVector, value_type, weight_type
export allocate_samples, rand_single

# buffers
export AbstractBuffer
export AbstractSampleBuffer
export SampleBuffer
export value_type, weight_type
export getsample, getsamples, setsample!, setsamples!
export getvalue, getvalues, setvalue!, setvalues!
export getweight, getweights, setweight!, setweights!


# utils
export filter_scan

# abstract sampler
export AbstractSampler

# proposal
export propose!
export UniformUnivariateProposal
export UniformProposal

# maximum finding
export NaiveMaxFinder, QuantileReductionMethod

# event generator
export EventGenerator
export input_type, output_type, proposal_distribution, target_distribution, maximum_value


using Distributions
using KernelAbstractions
using Atomix
using Adapt
using Random
using GPUArrays
using StaticArrays
using StaticArrays: sacollect
using StructArrays

include("patches/gpuarrays.jl")

include("samples/interface.jl")
include("samples/generic.jl")
include("samples/impl.jl")

include("buffers/interface.jl")
include("buffers/samplebuffer.jl")

include("filter_scan.jl")

include("proposal/random.jl")
include("proposal/interface.jl")
include("proposal/generics.jl")
include("proposal/uniform.jl")

include("sampler/interface.jl")
include("sampler/generics.jl")

include("target.jl")

include("generation/sampler.jl")
include("generation/buffers.jl")
include("generation/stages.jl")
include("generation/generate.jl")

# max finding
include("max_finder/types.jl")
include("max_finder/findmax.jl")
include("max_finder/naive.jl")
include("max_finder/quantile_reduction.jl")


include("plotting.jl")

include("testutils/TestUtils.jl")

include("mocks/Mocks.jl")

end
