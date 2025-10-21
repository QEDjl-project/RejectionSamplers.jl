RNG = Xoshiro(137137)

# TODO: update to proposal refac
function testsuite_univariate_generation(backend, vec_type, el_type, N, batch_size)
    mu = 5 * rand(RNG, el_type)                  # central value
    sig = rand(RNG, el_type)                     # variance
    dom = el_type.((mu - 0.5 * sig, mu + 3.0 * sig))   # support/domain

    ## example distribution to be sampled
    dist = TruncatedGaussian1D(mu, sig, dom)
    proposal = UniformUnivariateProposal(dom)

    ## max value
    max_value = maximum_value(dist)

    EG = EventGenerator(dist, proposal, max_value; backend, in_type = el_type, out_type = el_type)

    data = RejectionSamplers.generate_events(
        EG,
        N,
        batch_size,
    )

    @testset "shape and type preservation" begin
        @test size(data[1]) == (N,)
        @test size(data[2]) == (N,)
        @test eltype(data[1]) == el_type # for TruncatedGaussian1D
        @test eltype(data[2]) == el_type
    end

    return nothing
end

# TODO: update to proposal refac
function testsuite_multivariate_generation(backend, vec_type, el_type, N, batch_size)

    DIMS = (1, rand(RNG, 2:5))
    @testset "dim = $dim" for dim in DIMS
        mu = Tuple(5 .* rand(RNG, el_type, dim))                  # central value
        sig = Tuple(rand(RNG, el_type, dim))                     # variance
        low = mu .- el_type(0.5) .* sig
        up = mu .+ el_type(3.0) .* sig

        ## example distribution to be sampled
        dist = TruncatedGaussian(mu, sig, low, up)
        proposal = UniformProposal(low, up)

        ## max value
        max_value = maximum_value(dist)

        EG = EventGenerator(dist, proposal, max_value; backend, in_type = SVector{dim, el_type}, out_type = el_type)

        data = RejectionSamplers.generate_events(
            EG,
            N,
            batch_size,
        )

        @testset "shape and type preservation" begin
            @test size(data[1]) == (N,)
            @test size(data[2]) == (N,)
            @test eltype(data[1]) == SVector{dim, el_type}
            @test eltype(data[2]) == el_type
        end
    end

    return nothing
end

function testsuite_buffer_allocation(backend, in_type, out_type, batch_size, res_size)
    target = MockTarget()
    proposal = MockProposal(in_type)
    max_val = rand(RNG, out_type)
    EG = EventGenerator(target, proposal, max_val; backend, in_type, out_type)

    @testset "batch buffer" begin
        buffer_direct = RejectionSamplers._allocate_batch_buffer(backend, in_type, out_type, batch_size)
        buffer_event_generator = RejectionSamplers._allocate_batch_buffer(EG, batch_size)

        @testset "$buffer_name" for (buffer_name, buffer) in (
                ("direct", buffer_direct),
                ("event_generator", buffer_event_generator),
            )
            @testset "sizes" begin
                @test size(buffer.args) == (batch_size,)
                @test size(buffer.probs) == (batch_size,)
                @test size(buffer.vals) == (batch_size,)
            end

            @testset "types" begin
                @test eltype(buffer.args) == in_type
                @test eltype(buffer.probs) == out_type
                @test eltype(buffer.vals) == out_type
            end
        end

    end

    @testset "output buffer" begin
        buffer_direct = RejectionSamplers._allocate_output_buffer(backend, in_type, out_type, batch_size, res_size)
        buffer_event_generator = RejectionSamplers._allocate_output_buffer(EG, batch_size, res_size)

        @testset "$buffer_name" for (buffer_name, buffer) in (
                ("direct", buffer_direct),
                ("event_generator", buffer_event_generator),
            )
            @testset "sizes" begin
                @test size(buffer.args) == (batch_size + res_size,)
                @test size(buffer.vals) == (batch_size + res_size,)
                @test size(buffer.current_size) == (1,)
            end

            @testset "types" begin
                @test eltype(buffer.args) == in_type
                @test eltype(buffer.vals) == out_type
                @test eltype(buffer.current_size) == UInt32 # fix for KA
            end

            @testset "Counter" begin
                @test Vector(buffer.current_size)[1] == 0
            end
        end

    end
    return nothing
end

function testsuite_proposal_stage(backend, vec_type, in_type, out_type, batch_size)

    target = MockTarget()
    proposal = MockProposal(in_type)
    max_val = rand(RNG, out_type)

    EG = EventGenerator(target, proposal, max_val; backend, in_type, out_type)

    buffer = RejectionSamplers._allocate_batch_buffer(EG, batch_size)

    RejectionSamplers.generate_proposals!(EG, buffer)

    # building groundtruth
    # NOTE: works, because the mock proposal is deterministic
    h_sample_groundtruth = Vector{in_type}(undef, batch_size)
    h_weight_groundtruth = Vector{out_type}(undef, batch_size)
    d_sample_groundtruth = vec_type(h_sample_groundtruth)
    d_weight_groundtruth = vec_type(h_weight_groundtruth)
    propose!(proposal, d_sample_groundtruth, d_weight_groundtruth; backend)

    @test _isapprox(
        Vector(buffer.args),
        Vector(d_sample_groundtruth)
    )

    @test _isapprox(
        Vector(buffer.vals),
        Vector(d_weight_groundtruth)
    )
    return nothing
end

# TODO: implement this if the resp. interface is implemented
function testsuite_probs_stage(backend, in_type, out_type, batch_size)
    # 1. build mock rng (deterministic and simple)
    # 2. init batch buffer
    # 3. call generate_probability!
    # 4. check if buffer is updated accordingly (using reproducable rng)
end

function testsuite_target_stage(backend, vec_type, in_type, out_type, batch_size)
    target = MockTarget()
    proposal = MockProposal(in_type)
    max_val = rand(RNG, out_type)

    EG = EventGenerator(target, proposal, max_val; backend, in_type, out_type)

    buffer = RejectionSamplers._allocate_batch_buffer(EG, batch_size)

    RejectionSamplers.generate_proposals!(EG, buffer)

    RejectionSamplers.evaluate_target!(EG, buffer)


    h_groundtruth = Vector{out_type}(undef, batch_size)
    d_groundtruth = vec_type(h_groundtruth)
    RejectionSamplers._compute!(target, d_groundtruth, buffer.args)

    @test isapprox(
        Vector(buffer.vals),
        Vector(d_groundtruth)
    )

    return nothing
end

#TODO: move these function to mocks/utils.jl

function mock_make_invalid!(buffer::RejectionSamplers.EventOutputBuffers{T, U}) where {T, U}
    return fill!(buffer.vals, -one(U))
end

function mock_generate_probabilities(buffer::RejectionSamplers.EventBatchBuffers{T, U}) where {T, U}
    return fill!(buffer.probs, U(0.5))
end

mock_max_value(::Type{T}) where {T <: Real} = one(T)
mock_max_value(::Type{SVector{D, T}}) where {D, T <: Real} = T(D)
mock_max_value(::Type{NTuple{D, T}}) where {D, T <: Real} = T(D)


mock_no_accepted(batch_size::Int) = isodd(batch_size) ? (batch_size + 1) / 2 : batch_size / 2

function testsuite_filterscan_stage(backend, vec_type, in_type, out_type, batch_size, res_size)
    target = MockTarget()
    proposal = MockProposal(in_type)
    max_val = mock_max_value(in_type)

    EG = EventGenerator(target, proposal, max_val; backend, in_type, out_type)

    batch_buffer = RejectionSamplers._allocate_batch_buffer(EG, batch_size)

    RejectionSamplers.generate_proposals!(EG, batch_buffer)
    RejectionSamplers.evaluate_target!(EG, batch_buffer)
    mock_generate_probabilities(batch_buffer)

    output_buffer = RejectionSamplers._allocate_output_buffer(EG, batch_size, res_size)
    mock_make_invalid!(output_buffer)
    no_accepted_groundtruth = mock_no_accepted(batch_size)

    RejectionSamplers.rejection_filter!(EG, batch_buffer, output_buffer)


    out_weights = Vector(output_buffer.vals)
    no_accepted = count(x -> x == one(out_type), out_weights)
    @test no_accepted == no_accepted_groundtruth
    @test Vector(output_buffer.current_size)[1] == no_accepted_groundtruth

    return nothing
end

# TODO: implement this if the probability generator is implemented
function testsuite_generate_batch(backend, in_type, out_type, batch_size, N)
    # 1. build mocks for proposal, target, probs and filterscan
    # 2. init batch buffer
    # 3. call generate_event_batch!
    # 4. check if batch and output buffer are updated accordingly
end

# TODO: implement this if the probability generator is implemented
function testsuite_generation(backend, in_type, out_type, batch_size, N)
    # 1. build mocks for proposal, target, probs and filterscan
    # 2. call hotloop only
    # 3. verify monotonic growth of the out_size
    # 4. call generate_events
    # 5. check if output reproduces the known pattern
end

function testsuite_sampler(backend, in_type, out_type)
    target = MockTarget()
    proposal = MockProposal(in_type)
    max_val = rand(RNG, out_type)

    EG = EventGenerator(target, proposal, max_val; backend, in_type, out_type)

    @test target_distribution(EG) == target
    @test proposal_distribution(EG) == proposal
    @test maximum_value(EG) == max_val
    @test get_backend(EG) == backend
    @test input_type(EG) == in_type
    return @test output_type(EG) == out_type
end
