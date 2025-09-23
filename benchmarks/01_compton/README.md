# GPUEventGenerators Benchmarks

This directory contains a benchmark suite for **[GPUEventGenerators.jl](https://github.com/...)**, designed to measure the performance of event generation on multiple compute backends.
Supported backends include **CUDA, oneAPI, AMDGPU, Metal, and CPU** (OpenCL may be added in the future).

## Installation

Clone the repository and instantiate the benchmark environment:

```bash
git clone https://codebase.helmholtz.cloud/qedjl-applications/GPUEventGenerators.jl.git
cd benchmarks/01_Compton
julia --project=. init.jl
```

(Optional) Install the GPU backends you want to use, for example:

```bash
cd benchmarks/01_Compton
julia --project -e 'using Pkg; Pkg.add("CUDA")'       # NVIDIA GPUs
julia --project -e 'using Pkg; Pkg.add("AMDGPU")'     # AMD GPUs
julia --project -e 'using Pkg; Pkg.add("oneAPI")'     # Intel GPUs/CPUs
julia --project -e 'using Pkg; Pkg.add("Metal")'      # Apple M-series GPUs
```

The requested backend will be installed automatically, but preinstallation decreases time
for the benchmark runs.

## Running Benchmarks

Use `run.jl` to run the benchmarks. The backend can be selected using one of the flags:

- `--CUDA` – NVIDIA GPUs (CUDA.jl)
- `--oneAPI` – Intel GPUs/CPUs (oneAPI.jl)
- `--AMDGPU` – AMD GPUs (AMDGPU.jl)
- `--Metal` – Apple GPUs (Metal.jl)
- `--CPU` – CPU fallback (no GPU acceleration)

Example usage:

```bash
# Go into the benchmark folder
cd benchmarks/01_Compton

# Run on CPU
julia --project run.jl --CPU

# Run on NVIDIA GPU
julia --project run.jl --CUDA
```

You can also run a subset of benchmarks by passing their names as additional arguments.
For example:

```bash
julia --project run.jl --CUDA hotloop
```

This will run only benchmarks starting with `hotloop`.

## Results

The benchmark results are stored in JSON files under `benchmarks/data/`.
The filenames follow the pattern:

```bash
bench_<backend>.json
```

where `<backend>` is one of `CPU`, `CUDA`, `oneAPI`, `AMDGPU`, `Metal`.

## Plotting

To visualize benchmark results, use `plot.jl`:

```bash
cd benchmarks
julia --project plot.jl
```

This generates plots in the `benchmarks/plots/` directory, comparing the different benchmarks and backends.

## Adding New Benchmarks

You can extend the benchmark suite by adding new tests. Follow these steps:

### Create a new benchmark file

Place a new `.jl` file in the `benchmarks/` directory. Name it descriptively, e.g., `my_new_benchmark.jl`.

### Define a benchmark group

Inside the file, define your benchmarks using `BenchmarkGroup`:

```julia

group = addgroup!(SUITE, "my_bench")

group["my_bench_1"] = @benchmarkable begin
    # Your benchmarked code here
end
```

This ensures your benchmark integrates with the suite’s warmup, tuning, and result collection.

### Optional: Specify supported backends

If your benchmark only works on certain backends, you can check the `BACKEND` constant:

```julia
if BACKEND isa CUDABackend
    # GPU-specific benchmark
end
```

### Run the benchmark

After adding your file, run `run.jl` as usual:

```bash
julia --project run.jl --CPU
```

The suite will automatically detect your new benchmark and include it.

### Profiling

To profile using ncu, use:

```bash
ncu -o cuda_profile --set full julia --project=. profile.jl --CUDA
```

`-o` sets the output file name, `--set full` turns on all the metrics. Then use the resulting file and open it in `ncu-ui` to inspect the kernel performance.
