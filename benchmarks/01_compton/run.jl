include("utils.jl")

benchmark_all = tryparse(Bool, get(ENV, "BENCH_ALL", "0"))
cpu_benchmarks = tryparse(Bool, get(ENV, "BENCH_CPU", "0")) || benchmark_all
cuda_benchmarks = tryparse(Bool, get(ENV, "BENCH_CUDA", "0")) || benchmark_all
amdgpu_benchmarks = tryparse(Bool, get(ENV, "BENCH_AMDGPU", "0")) || benchmark_all
oneapi_benchmarks = tryparse(Bool, get(ENV, "BENCH_ONEAPI", "0")) || benchmark_all
metal_benchmarks = tryparse(Bool, get(ENV, "BENCH_METAL", "0")) || benchmark_all

if cpu_benchmarks
    @safebenchmark "CPUBenchmark" begin
        include("benchmarks/benchmark_cpu.jl")
    end
end

if cuda_benchmarks
    @safebenchmark "CUDABenchmark" begin
        include("benchmarks/benchmark_cuda.jl")
    end
end

if amdgpu_benchmarks
    @safebenchmark "AMDGPUBenchmark" begin
        include("benchmarks/benchmark_amdgpu.jl")
    end
end

if oneapi_benchmarks
    @safebenchmark "oneAPIBenchmark" begin
        include("benchmarks/benchmark_oneapi.jl")
    end
end

if metal_benchmarks
    @safebenchmark "MetalBenchmark" begin
        include("benchmarks/benchmark_metal.jl")
    end
end

any_benchmark =
    cpu_benchmarks ||
    cuda_benchmarks ||
    amdgpu_benchmarks ||
    oneapi_benchmarks ||
    metal_benchmarks
if !any_benchmark
    @warn "no benchmarks were performed!"
end
