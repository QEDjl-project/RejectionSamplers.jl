# Example: Constrained Gaussian

This repository demonstrates how to generate samples from a **truncated Gaussian distribution**,
a non-trivial extension of the Gaussian distribution confined to a specified interval.

## Features

- sampling of truncated Gaussian distributions.
- Comparison of generated samples with the target distribution using histograms.

---

## Setup Instructions

### 1. Clone the Repository

Ensure you have cloned the repository containing this example.

```bash
git clone git@github.com/qedjl-project/RejectionSamplers.jl.git
cd <repository-directory>/example
```

### 2. Initialize the Example Environment

Since `RejectionSamplers.jl` is not a registered Julia package,
you need to initialize the environment manually. Run the following command inside the `examples/truncated_gaussian` directory:

```bash
julia --project=. init.jl
```

This script will:

- Activate `RejectionSamplers.jl` in the example environment.
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
- The histogram is saved as `example_plot_compare.pdf` in the `examples/truncated_gaussian` directory.

## Requirements

Ensure you have the following tools installed:

- **Julia**: Version 1.10 or later is recommended.

## Troubleshooting

- **Dependency Issues**: If you encounter problems during initialization, verify that the `init.jl` script executed successfully and installed all required packages.
