
import Pkg

main_package_jl = Pkg.PackageSpec(path = "../../")

Pkg.develop(main_package_jl)
Pkg.instantiate()
