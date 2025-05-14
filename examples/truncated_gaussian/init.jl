import Pkg

main_package_jl = Pkg.PackageSpec(path = "../../")
truncated_gaussians_jl = Pkg.PackageSpec(path = "./TruncatedGaussians")

Pkg.develop([main_package_jl, truncated_gaussians_jl])
Pkg.instantiate()
