RNG = Xoshiro(137137)

function testsuite_univariate_generation(backend, vec_type, el_type, N, batch_size)
    mu = 5 * rand(RNG, el_type)                  # central value
    sig = rand(RNG, el_type)                     # variance
    dom = el_type.((mu - 0.5 * sig, mu + 3.0 * sig))   # support/domain

    ## example distribution to be sampled
    dist = TruncatedGaussian1D(mu, sig, dom)
    proposal = UniformUnivariateProposal(dom)

    ## max value
    max_value = maximum_value(dist)

    data = GPUEventGenerators.generate_events(
        dist,
        proposal,
        max_value,
        N,
        batch_size,
        backend,
        el_type,
    )

    return @testset "shape and type preservation" begin
        @test size(data[1]) == (N,)
        @test size(data[2]) == (N,)
        @test eltype(data[1]) == el_type # for TruncatedGaussian1D
        @test eltype(data[2]) == el_type
    end
end

function testsuite_multivariate_generation(backend, vec_type, el_type, N, batch_size)

    DIMS = (1, rand(RNG, 2:5))
    return @testset "dim = $dim" for dim in DIMS
        mu = Tuple(5 .* rand(RNG, el_type, dim))                  # central value
        sig = Tuple(rand(RNG, el_type, dim))                     # variance
        low = mu .- el_type(0.5) .* sig
        up = mu .+ el_type(3.0) .* sig

        ## example distribution to be sampled
        dist = TruncatedGaussian(mu, sig, low, up)
        proposal = UniformMultivariateProposal(low, up)

        ## max value
        max_value = maximum_value(dist)

        data = GPUEventGenerators.generate_events(
            dist,
            proposal,
            max_value,
            N,
            batch_size,
            backend,
            el_type,
            SVector{dim, el_type},
        )

        @testset "shape and type preservation" begin
            @test size(data[1]) == (N,)
            @test size(data[2]) == (N,)
            @test eltype(data[1]) == SVector{dim, el_type}
            @test eltype(data[2]) == el_type
        end
    end
end

function testsuite_buffer_allocation(backend, dtype, out_dtype, N, batch_size)
    @testset "batch buffer" begin
        buffer = GPUEventGenerators._allocate_batch_buffer(backend, dtype, out_dtype, batch_size)

        @testset "sizes" begin
            @test size(buffer.args) == (batch_size,)
            @test size(buffer.probs) == (batch_size,)
            @test size(buffer.vals) == (batch_size,)
        end

        @testset "types" begin
            @test eltype(buffer.args) == dtype
            @test eltype(buffer.probs) == out_dtype
            @test eltype(buffer.vals) == out_dtype
        end

    end

    return @testset "output buffer" begin
        buffer = GPUEventGenerators._allocate_output_buffer(backend, dtype, out_dtype, batch_size, N)

        @testset "sizes" begin
            @test size(buffer.args) == (batch_size + N,)
            @test size(buffer.vals) == (batch_size + N,)
            @test size(buffer.current_size) == (1,)
        end

        @testset "types" begin
            @test eltype(buffer.args) == dtype
            @test eltype(buffer.vals) == out_dtype
            @test eltype(buffer.current_size) == UInt32 # fix for KA
        end

        @testset "Counter" begin
            @test buffer.current_size[1] == 0
        end

    end
end

function testsuite_proposal_stage(backend, dtype, out_dtype, batch_size)
    # 1. build mock proposal based on dtype and some dimension (deterministic + simple)
    # 2. init batch buffer
    # 3. call generate_proposal!
    # 4. check if buffer is updated accordingly (using reproducable proposal)
end

function testsuite_probs_stage(backend, dtype, out_dtype, batch_size)
    # 1. build mock rng (deterministic and simple)
    # 2. init batch buffer
    # 3. call generate_probability!
    # 4. check if buffer is updated accordingly (using reproducable rng)
end

function testsuite_target_stage(backend, dtype, out_dtype, batch_size)
    # 1. build mock target (deterministic and simple)
    # 2. init batch buffer
    # 3. call evaluate_target!
    # 4. check if buffer is updated accordingly
end

function testsuite_filterscan_stage(backend, dtype, out_dtype, batch_size, N)
    # 1. build mock arrays for the buffer
    # 2. build known pattern into vals and probs, so the rejection pattern is
    # deterministic
    # 3. init batch and output buffer
    # 4. call filter_scan
    # 5. check if output buffer is updated accordingly
end

function testsuite_generate_batch(backend, dtype, out_dtype, batch_size, N)
    # 1. build mocks for proposal, target, probs and filterscan
    # 2. init batch buffer
    # 3. call generate_event_batch!
    # 4. check if batch and output buffer are updated accordingly
end

function testsuite_generation(backend, dtype, out_dtype, batch_size, N)
    # 1. build mocks for proposal, target, probs and filterscan
    # 2. call hotloop only
    # 3. verify monotonic growth of the out_size
    # 4. call generate_events
    # 5. check if output reproduces the known pattern
end

# TODO:
# - fill the functions
# - run the testsuite above for all el_types and dtype=(el_type,NTuple,SVector,PSP ?)
