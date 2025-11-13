using Documenter

#push!(LOAD_PATH,"../src/")

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
)

deploydocs(
    repo = "github.com/MPF-Optimization-Laboratory/BlockTensorFactorization.jl.git",
)
