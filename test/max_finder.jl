RNG = Xoshiro(137)

function testsuite_max_finder_gaussian(backend, vec_type, out_dtype, dim)

    if !(backend isa CPU)
        # max finding currently only supports CPU backend
        @test_broken false
        return nothing
    end


    mu = ntuple(_ -> rand(RNG, out_dtype) * 4 .- out_dtype(2.0), dim)                # values roughly in [-2, 2]
    sig = ntuple(_ -> rand(RNG, out_dtype) * 2 .+ out_dtype(0.5), dim)              # values roughly in [0.5, 2.5]

    lower = ntuple(i -> -out_dtype(3.0) * sig[i], dim)
    upper = ntuple(i -> mu[i] + out_dtype(3.0) * sig[i], dim)
    dom = (lower, upper)


    dist = TruncatedGaussian(mu, sig, dom...)
    proposal = UniformProposal(dom...)
    groundtruth = maximum_value(dist)

    @testset "Naive max finder" begin
        max_finder = NaiveMaxFinder(Int(1.0e7))
        max_val = findmax(
            RNG,
            out_dtype,
            dist,
            proposal,
            max_finder;
            dtype = SVector{dim, out_dtype},
        )

        @test isapprox(groundtruth, max_val, atol = 1.0e-2)
    end

    return @testset "quantile reduction" begin
        p = out_dtype(0.001) # quantile which is ignored
        max_finder = QuantileReductionMethod(p, Int(1.0e6))
        max_val = findmax(
            RNG,
            out_dtype,
            dist,
            proposal,
            max_finder;
            dtype = SVector{dim, out_dtype},
        )


        #@test isapprox(groundtruth*(1-p),max_val,atol=1e-4)

        @testset "n: %n" for n in (Int(2.0e5), Int(1.0e6))

            #FIXME: quantile estimate seems broken for F16
            if !(out_dtype == Float16)
                # groundtruth
                samples = Vector{SVector{dim, out_dtype}}(undef, n)
                weights = Vector{out_dtype}(undef, n)
                propose!(RNG, proposal, samples, weights; backend)
                weights = sort(RejectionSamplers._compute.(dist, samples))
                residual_weights = @. max(1, weights / max_val)
                idx_last_unit_weight =
                    length(residual_weights) -
                    length(residual_weights[residual_weights .> 1.0])

                test_quantile =
                    1.0 -
                    sum(residual_weights[1:idx_last_unit_weight]) / sum(residual_weights)

                @test p >= test_quantile
            end
        end

    end
end
