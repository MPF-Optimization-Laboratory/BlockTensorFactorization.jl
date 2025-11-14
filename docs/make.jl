using Documenter

# push!(LOAD_PATH,"../")
# push!(LOAD_PATH,"../src/")

using BlockTensorFactorization

DocMeta.setdocmeta!(
    BlockTensorFactorization,
    :DocTestSetup,
    :(using BlockTensorFactorization;);
    recursive=true
)

makedocs(
    sitename="BlockTensorFactorization.jl",
    modules = [BlockTensorFactorization,],
    checkdocs=:exports,
    pages = [
        "Home" => "index.md",
        "Quick Guide" => "quickguide.md",
        "Tutorial" => [
            "tutorial/decompositionmodels.md",
            "tutorial/constraints.md",
            "tutorial/blockupdateorder.md",
            "tutorial/iterationstats.md",
        ],
        "Reference" => [
            "reference/types.md",
            "reference/functions.md",
            "reference/index.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/BlockTensorFactorization.jl.git",
)
