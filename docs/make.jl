using GPUEventGenerators
using Documenter

DocMeta.setdocmeta!(
    GPUEventGenerators,
    :DocTestSetup,
    :(using GPUEventGenerators);
    recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [GPUEventGenerators],
    authors = "Uwe Hernandez Acosta <u.hernandez@hzdr.de>, Simeon Ehrig, Anton Reinhard, René Widera",
    repo = "https://github.com/QEDjl-project/GPUEventGenerators.jl/blob/{commit}{path}#{line}",
    sitename = "GPUEventGenerators.jl",
    format = Documenter.HTML(;
        canonical = "https://QEDjl-project.github.io/GPUEventGenerators.jl",
    ),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/QEDjl-project/GPUEventGenerators.jl")
