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
    p = Lp.p
    return norm(X - Y; p)^p # TODO should this be 1/p ? I think not since p could be Inf
end