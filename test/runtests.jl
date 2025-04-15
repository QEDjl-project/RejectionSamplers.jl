using GPUEventGenerators
using Test

@testset "GPUEventGenerators.jl" begin
    @test GPUEventGenerators.hello_world() == "Hello, World!"
end
