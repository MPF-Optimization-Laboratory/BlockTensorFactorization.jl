"""
AbstractObjective <: Function

General interface is

struct L2 <: AbstractObjective end

after constructing

myobjective = L2()

you can call

myobjective(X, Y)
"""
abstract type AbstractObjective <: Function end

"""
    L2 <: AbstractObjective

The least squares objective.
"""
struct L2 <: AbstractObjective end

"""
    (objective::L2)(X, Y)

Calculates the least squares objective at tensors `X` and `Y`.
"""
(objective::L2)(X, Y) = norm2(X - Y)

# TODO Should this be 0.5norm2(X - Y) instead?

"""
    L1 <: AbstractObjective

The absolute error objective.
"""
struct L1 <: AbstractObjective end

"""
    (objective::L1)(X, Y)

Calculates the absolute error objective at tensors `X` and `Y`.
"""
(objective::L1)(X, Y) = sum(abs, X - Y)

"""
    Lp <: AbstractObjective

The p-norm distance objective ||X-Y||_p^p.
"""
struct Lp <: AbstractObjective 
    p::Real
end

"""
    (objective::Lp)(X, Y)

Calculates the p-norm objective at tensors `X` and `Y`.
"""
function (objective::Lp)(X, Y)
    p = objective.p
    return norm(X - Y, p)^p # TODO should this be 1/p ? I think not since p could be Inf
end

"""
    KLDivergence <: AbstractObjective

KL-divergence: `sum_i X[i] ln (X[i] / Y[i])`. Should be used with simplex-like constraints.

This is the f-divergence with `f(t)=t*ln(t)`.
"""
struct KLDivergence <: AbstractObjective end

"""
    (KLDivergence::Lp)(X, Y)

Calculates the KL-divergence objective at tensors `X` and `Y`.
"""
function (objective::KLDivergence)(X, Y)
    return sum(@. abs(X) * log(abs(X / Y)))
end

"""
    AbstractStructuredObjective <: AbstractObjective

Abstract supertype for objectives that operate row, column, or slice-wise between arrays.
"""
abstract type AbstractStructuredObjective <: AbstractObjective end

"""
    SliceWiseObjective <: AbstractStructuredObjective
    SliceWiseObjective(objective, whats_compared)

Compares each slice of the arrays according to the objective, and sums the objectives.

The provided `objective` can be an instance that is an `AbstractObjective` (e.g. `L2()`)
or the `DataType` as along as it can be made without arguments (e.g `L2`).
"""
struct SliceWiseObjective{T<:AbstractObjective} <: AbstractStructuredObjective
    objective::T
    whats_compared::Function
end

function SliceWiseObjective(objective::AbstractObjective, whats_compared)
    SliceWiseObjective{typeof(objective)}(objective, whats_compared)
end

function SliceWiseObjective(objective::DataType, whats_compared)
    instance = objective() # will error if arguments are needed (e.g. Lp(p) needs a `p`)
    SliceWiseObjective{objective}(instance, whats_compared)
end

"""
    (objective::SliceWiseObjective)(X, Y)

Calculated the objective between each slice of X and Y.
"""
(T::SliceWiseObjective)(X, Y) = sum(T.objective(x, y) for (x, y) in zip(T.whats_compared(X), T.whats_compared(Y)))

# """RowWiseObjective(objective) = SliceWiseObjective(objective, eachrow)"""
# RowWiseObjective(objective) = SliceWiseObjective(objective, eachrow)
# """ColWiseObjective(objective) = SliceWiseObjective(objective, eachcol)"""
# ColWiseObjective(objective) = SliceWiseObjective(objective, eachcol)
# """Slice1WiseObjective(objective) = SliceWiseObjective(objective, x -> eachslice(x; dims=1))"""
# Slice1WiseObjective(objective) = SliceWiseObjective(objective, x -> eachslice(x; dims=1))
# """Slice12WiseObjective(objective) = SliceWiseObjective(objective, x -> eachslice(x; dims=(1,2)))"""
# Slice12WiseObjective(objective) = SliceWiseObjective(objective, x -> eachslice(x; dims=(1,2)))

names_and_slice = [
    "Row" => eachrow,
    "Col" => eachcol,
    "Slice1" => each1slice,
    "Slice12" => each12slice,
]

BUILT_IN_STRUCTURED_OBJECTIVES = Symbol[]

for (name, whats_compared) in names_and_slice
    function_name = "$(name)WiseObjective"
    function_name = Symbol(function_name)
    supertype = "SliceWiseObjective"

    definition = "$(supertype)(objective, $whats_compared)"
    # full_definition = "$function_name(objective) = $definition"
    docstring =
"""
    $(function_name)(objective)

Alias for

`$(definition)`.

See [`$(supertype)`](@ref).
"""
    eval(quote
        const $function_name($(:objective)) = SliceWiseObjective($(:objective), $whats_compared)
        @doc $docstring $function_name
    end)
    push!(BUILT_IN_STRUCTURED_OBJECTIVES, function_name)
end