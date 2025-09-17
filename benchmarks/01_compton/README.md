# GPUEventGenerators Benchmarks

This directory contains a benchmark suite for **[GPUEventGenerators.jl](https://github.com/...)**, designed to measure the performance of event generation on multiple compute backends.
Supported backends include **CUDA, oneAPI, AMDGPU, Metal, and CPU** (OpenCL may be added in the future).

## Installation

Clone the repository and instantiate the benchmark environment:

```bash
git clone https://codebase.helmholtz.cloud/qedjl-applications/GPUEventGenerators.jl.git
cd benchmarks
julia --project=. init.jl
```

(Optional) Install the GPU backends you want to use, for example:

```bash
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
# Run on CPU
julia --project=benchmarks benchmarks/run.jl --CPU

# Run on NVIDIA GPU
julia --project=benchmarks benchmarks/run.jl --CUDA
```

You can also run a subset of benchmarks by passing their names as additional arguments.
For example:

```bash
julia --project=benchmarks benchmarks/run.jl --CUDA hotloop
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
julia --project=. plot.jl
```

This generates plots in the `benchmarks/plots/` directory, comparing the different benchmarks and backends.

Benchmarks are located in the `benchmarks/` folder.
To add a new benchmark:

1. Create a new `.jl` file in `benchmarks/`.
2. Define a `BenchmarkGroup` entry for your test.
3. Run `run.jl` again — the new benchmark will be automatically discovered and included.
