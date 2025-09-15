import Pkg

@info "Performing cleanup."

PKGS = ["InteractiveUtils"]
for _pkg in PKGS
    _pkg in keys(Pkg.project().dependencies) ? Pkg.rm(_pkg) : nothing
end

Pkg.gc()
