using RejectionSamplers
using RejectionSamplers.TestUtils
using Documenter

DocMeta.setdocmeta!(
    RejectionSamplers,
    :DocTestSetup,
    :(using RejectionSamplers);
    recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
        file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [RejectionSamplers],
    authors = "Uwe Hernandez Acosta <u.hernandez@hzdr.de>, Simeon Ehrig, Anton Reinhard, René Widera",
    repo = Documenter.Remotes.GitHub("QEDjl-project", "RejectionSamplers.jl"),
    sitename = "RejectionSamplers.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        #repolink = "https://github.com/qedjl-project/RejectionSamplers.jl/blob/{commit}{path}#{line}",
        canonical = "https://qedjl-project.github.io/RejectionSamplers.jl",
        assets = String[],
    ),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/QEDjl-project/RejectionSamplers.jl", push_preview = false)
