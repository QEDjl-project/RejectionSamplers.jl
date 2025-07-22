using Pkg

# dev branches from QuantumElectrodynamics.jl

packages = (
    PackageSpec(name = "QEDbase", rev = "dev"),
    PackageSpec(name = "QEDcore", rev = "dev"),
    PackageSpec(name = "QEDprocesses", rev = "dev"),
)

Pkg.add.(packages)
