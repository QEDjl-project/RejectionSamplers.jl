# TODO: rename this file

# TODO: move this to separate file
SUPPORTED_IN_TYPES = Union{Float16, Float32, Float64}
OUT_TYPES_F64 = Union{Float64, SVector{N, Float64}, NTuple{N, Float64}} where {N}
OUT_TYPES_F32 = Union{Float32, SVector{N, Float32}, NTuple{N, Float32}} where {N}
OUT_TYPES_F16 = Union{Float16, SVector{N, Float16}, NTuple{N, Float16}} where {N}
SUPPORTED_OUT_TYPES = Union{OUT_TYPES_F16, OUT_TYPES_F32, OUT_TYPES_F64}

function _assert_compat_io_types(::Type{IN_T}, ::Type{OUT_T}) where {IN_T, OUT_T}
    throw(
        ArgumentError(
            "input type and output type must be compatible"
        )
    )
end
_assert_compat_io_types(::Type{IN_T}, ::Type{OUT_T}) where {N, IN_T <: SUPPORTED_IN_TYPES, OUT_T <: Union{IN_T, SVector{N, IN_T}, NTuple{N, IN_T}}} = nothing

# TODO: implement proper compat assert
function _assert_compat_target(target, proposal, in_type, out_type) end

_assert_compat_backend(backend::Backend, ::Type{IN_T}, ::Type{OUT_T}) where {IN_T <: SUPPORTED_IN_TYPES, OUT_T <: SUPPORTED_OUT_TYPES} = nothing
function _assert_compat_backend(backend::Backend, ::Type{Float64}, ::Type{OUT_T}) where {OUT_T <: OUT_TYPES_F64}
    return KernelAbstractions.supports_float64(backend)
end


abstract type AbstractRejectionSampler end

struct EventGenerator{IN_T, OUT_T, TARGET, PROPOSAL, BACKEND} <: AbstractRejectionSampler
    target::TARGET
    proposal::PROPOSAL
    max_value::IN_T
    backend::BACKEND

    function EventGenerator(
            target::TARGET,
            proposal::PROPOSAL,
            max_val::IN_T;
            backend::BACKEND = CPU(), # default backend
            in_type::Type{IN_T},
            out_type::Type{OUT_T}
        ) where {
            IN_T <: SUPPORTED_IN_TYPES,
            OUT_T <: SUPPORTED_OUT_TYPES,
            # change to AbstractScatteringProcessDistribution if available
            TARGET <: AbstractTargetDistribution,
            PROPOSAL <: AbstractProposal, #AbstractCoordinateProposal{OUT_T},
            BACKEND <: KernelAbstractions.Backend,
        }

        _assert_compat_io_types(in_type, out_type)
        _assert_compat_target(target, proposal, in_type, out_type)
        _assert_compat_backend(backend, in_type, out_type)

        return new{IN_T, OUT_T, TARGET, PROPOSAL, BACKEND}(target, proposal, max_val, backend)
    end
end

input_type(::EventGenerator{IN_T}) where {IN_T} = IN_T
output_type(::EventGenerator{IN_T, OUT_T}) where {IN_T, OUT_T} = OUT_T
proposal_distribution(eg::EventGenerator) = eg.proposal
target_distribution(eg::EventGenerator) = eg.target
maximum_value(eg::EventGenerator) = eg.max_value
KernelAbstractions.get_backend(eg::EventGenerator) = eg.backend
