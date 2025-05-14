# Example: Constrained Gaussian

This repository demonstrates how to generate samples from a **truncated Gaussian distribution**,
a non-trivial extension of the Gaussian distribution confined to a specified interval.
The distribution and sampling functionality (CPU-based) are implemented in the `TruncatedGaussians.jl` package.

## Features

- sampling of truncated Gaussian distributions.
- Comparison of generated samples with the target distribution using histograms.

---

## Setup Instructions

### 1. Clone the Repository

Ensure you have cloned the repository containing this example.

```bash
git clone git@codebase.helmholtz.cloud:qedjl-applications/GPUEventGenerators.jl.git
cd <repository-directory>/example
```

### 2. Initialize the Example Environment

Since `GPUEventGenerators.jl` and `TruncatedGaussians.jl` are not registered Julia packages,
you need to initialize the environment manually. Run the following command inside the `example` directory:

```bash
julia --project=. init.jl
```

This script will:

- Link the source code of `GPUEventGenerators.jl` and `TruncatedGaussians.jl` to the example environment.
- Install all other necessary dependencies.
- Instantiate the example environment

---

## Running the Example

### 1. Execute the Example Script

To generate samples and visualize the constrained Gaussian distribution, execute the following command inside the `example` directory:

```bash
julia --project example.jl
```

### 2. Output

- The script generates **1 million samples** from a constrained Gaussian distribution.
- A histogram comparing the generated samples with the target distribution is plotted.
- The histogram is saved as `example_plot_compare.pdf` in the `example` directory.

### 3. Interactive Exploration (Optional)

For interactive experimentation, open the Jupyter notebook `example.iypnb`:

1. Ensure you have the `IJulia` package installed:

   ```julia
   using Pkg
   Pkg.add("IJulia")
   ```

2. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook
   ```

This allows you to explore the constrained Gaussian example interactively using Julia's kernel.

---

## Requirements

Ensure you have the following tools installed:

- **Julia**: Version 1.10 or later is recommended.
- **Jupyter** (optional): For interactive notebooks.

---

## Troubleshooting

- **Dependency Issues**: If you encounter problems during initialization, verify that the `init.jl` script executed successfully and installed all required packages.
- **Missing `IJulia`**: Ensure the `IJulia` package is installed to enable Julia kernels in Jupyter.
