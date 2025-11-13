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
