# RejectionSamplers.jl

[![Doc Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://qedjl-project.github.io/RejectionSamplers.jl/stable)
[![Doc Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://qedjl-project.github.io/RejectionSamplers.jl/dev)
[![pipeline status](https://gitlab.com/hzdr/qedjl-project/RejectionSamplers-jl/badges/dev/pipeline.svg)](https://gitlab.com/hzdr/qedjl-project/RejectionSamplers-jl/-/commits/dev)
[![codecov](https://codecov.io/gh/QEDjl-project/RejectionSamplers.jl/graph/badge.svg?token=46ATK29J1F)](https://codecov.io/gh/QEDjl-project/RejectionSamplers.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

**RejectionSamplers.jl** provides a flexible, hardware-agnostic framework for implementing
and executing rejection sampling algorithms in Julia. The package separates the core
stages of the rejection sampling process into independent, extensible interfaces, making
it suitable for a wide range of applications, This includes simple numerical sampling, but
also full-scale Monte-Carlo event generation.

## Key Features

- **Hardware-agnostic design:** runs on both CPU and GPU backends via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
- **Composable interfaces:** each stage of the rejection algorithm has its own abstract interface:

  - **Proposal generation:** defines how candidate samples are drawn.
  - **Target evaluation:** specifies the target distribution.
  - **Probability generation:** computes acceptance probabilities.
  - **Rejection filtering:** performs the acceptance/rejection step.

- **Modular and extensible:** each stage can be specialized for specific domains, data types, or hardware architectures.
- **Efficient parallel execution:** suitable for large-scale Monte-Carlo workloads or GPU-accelerated applications.

## Installation

`RejectionSamplers.jl` can be installed via Julia’s package manager:

```julia
julia> import Pkg
julia> Pkg.add("RejectionSamplers")
```

You can also use the Pkg REPL mode by pressing `]` and then running:

```julia
pkg> add RejectionSamplers
```

If you'd like to use the development version instead:

```julia
pkg> add RejectionSamplers#dev
```

Or clone manually and instantiate:

```bash
git clone https://github.com/QEDjl-project/RejectionSamplers.jl.git
cd RejectionSamplers.jl
julia --project=@.
julia> ] instantiate
```

## Contributing

Contributions are welcome! If you'd like to report a bug, suggest an enhancement, or contribute
code, please feel free to open an issue or submit a pull request.

To ensure consistency across the `QuantumElectrodynamics.jl` ecosystem, we encourage all contributors
to review the [QuantumElectrodynamics.jl contribution guide](https://qedjl-project.github.io/QuantumElectrodynamics.jl/stable/dev_guide/#Development-Guide).

## Credits and contributors

This work was partly funded by the Center for Advanced Systems Understanding (CASUS) that
is financed by Germany’s Federal Ministry of Education and Research (BMBF) and by the Saxon
Ministry for Science, Culture and Tourism (SMWK) with tax funds on the basis of the budget
approved by the Saxon State Parliament.

The core code of the package `RejectionSamplers.jl` is developed by a small team at the Center for
Advanced Systems Understanding ([CASUS](https://www.casus.science)) and the
Helmholtz-Zentrum Dresden Rossendorf ([HZDR](http://www.hzdr.de)), namely

### Contributors

- **Uwe Hernandez Acosta** (CASUS/HZDR, [u.hernandez@hzdr.de](mailto:u.hernandez@hzdr.de))
- **Anton Reinhard** (CASUS/HZDR)
- **Simeon Ehrig** (CASUS/HZDR)
- **Klaus Steiniger** (CASUS/HZDR)
- **Rene Widera** (HZDR)

### Acknowledgements

We extend our gratitude for the support received through direct and indirect funding for this project, especially

- **Michael Bussmann**
- **Tobias Dornheim**

## License

[MIT](LICENSE) © Uwe Hernandez Acosta
