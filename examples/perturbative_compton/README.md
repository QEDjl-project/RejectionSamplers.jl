# Example: Klein-Nishina

This repository demonstrates how to generate samples from a **Klein-Nishina distribution**,
the solid angle distribution of the scattering of a single photon off an electron at rest.

## Features

- sampling of Klein-Nishina distribution.
- Comparison of generated samples with the target distribution using histograms.

---

## Setup Instructions

### 1. Clone the Repository

Ensure you have cloned the repository containing this example.

```bash
git clone git@github.com/qedjl-project/RejectionSamplers.jl.git
cd <repository-directory>/examples/perturbative_compton
```

### 2. Initialize the Example Environment

Since `RejectionSamplers.jl` is not registered Julia packages,
you need to initialize the environment manually. Run the following command inside the `examples/perturbative_compton` directory:

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

To generate samples and visualize the Klein-Nishina distribution, execute the following command inside the `examples/perturbative_compton` directory:

```bash
julia --project example.jl
```

### 2. Output

- The script generates **1 million samples** from a Compton distribution.
- A histogram comparing the generated samples with the target distribution is plotted.
- The histogram is saved as `corner_plot_*.pdf` and `weights_*.pdf` in the `plots` directory.

## Requirements

Ensure you have the following tools installed:

- **Julia**: Version 1.10 or later is recommended.

---

## Troubleshooting

- **Dependency Issues**: If you encounter problems during initialization, verify that the `init.jl` script executed successfully and installed all required packages.
