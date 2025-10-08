using Pkg

# dev branches from QuantumElectrodynamics.jl

packages = (
    PackageSpec(name = "QEDbase", rev = "c3861d67767deb3449581c7e71f15a4b3e5f2724"),
    PackageSpec(name = "QEDcore", rev = "dev"),
    PackageSpec(name = "QEDprocesses", rev = "dev"),
)

Pkg.add.(packages)
