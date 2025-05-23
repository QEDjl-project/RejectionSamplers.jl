# GPUEventGenerators.jl

![Stable](https://img.shields.io/badge/docs-main-blue.svg)](<https://qedjl-applications.pages.hzdr.de/GPUEventGenerators.jl>)
[![pipeline status](https://codebase.helmholtz.cloud/qedjl-applications/GPUEventGenerators.jl/badges/main/pipeline.svg)](https://codebase.helmholtz.cloud/qedjl-applications/GPUEventGenerators.jl/-/commits/main)

`GPUEventGenerators.jl` is a showcase project for Monte-Carlo Event Generation
on GPU. Its main goal is the investigation of end-to-end workflows to generate events for
scattering processes.

> 🚧 **Warning**: This package is under rapid development. Expect **breaking changes without notice**.

## Installation

Since `GPUEventGenerators.jl` is not registered (and probably never will), you need to
clone the repository by youself:

### Clone with ssh (recommended)

```bash
git clone git@codebase.helmholtz.cloud:qedjl-applications/GPUEventGenerators.jl.git
```

### Clone with https

```bash
git clone https://codebase.helmholtz.cloud/qedjl-applications/GPUEventGenerators.jl.git
```

Within the root directory of the project, you can instantiate the project by entering the
Julia REPL with `julia --project=@.` and using `Pkg`:

```julia-repl
julia> # press ]
pkg> instantiate
```

## Usage

To use the package in your Julia code, simply import it:

```julia-repl
using GPUEventGenerators
```

For detailed documentation on available functions and examples, please refer to the
package documentation or source code.

## Running Tests

You can run the provided test suite to ensure that the package is functioning correctly.
You need to open your julia REPL within the project's directory

```bash
julia --project=@.
```

In the Julia REPL, you need to enter the pkg mode and run the tests:

```julia-repl
julia> # press ]
pkg> activate .
pkg> test
```

This will execute the test suite and display the results, indicating whether the package functions as expected.

## Formatting

We use [JuliaFormatter.jl](https://domluna.github.io/JuliaFormatter.jl/dev/) and the [Blue
style](https://github.com/invenia/BlueStyle) to format our code. The correct form of the
code is checked by a CI job. To format the code manually, run the following commands:

```bash
# install dependencies
julia --project=.formatting -e 'import Pkg; Pkg.instantiate()'
# format all documents
julia --project=.formatting .formatting/format_all.jl
```

## Building Docs Locally

Building the docs locally involves the following steps:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); Pkg.develop(PackageSpec(path=pwd()))'
julia --project=docs --color=yes docs/make.jl
```

The website with the documentation can then be accessed using the browser of your choice
by opening the file `docs/build/index.html`.
