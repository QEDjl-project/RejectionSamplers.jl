using Pkg

# dev branches from QuantumElectrodynamics.jl

packages = (
    #PackageSpec(name = "QEDbase"),
    #PackageSpec(name = "QEDcore"),
    PackageSpec(name = "QEDprocesses", rev = "dev"),
)

Pkg.add.(packages)
