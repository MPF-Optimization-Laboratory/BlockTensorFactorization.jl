"""
AbstractStep and structure to calculate and store stepsizes for descent
"""

"""
Interface to make a step scheme is

```
struct MyStep <: AbstractStep
    ...
end

function (step::MyStep)(x::AbstractDecomposition; kwargs...)
    ...
    return step::Real
end
```

To use your scheme, construct an instance with any necessary parameters

`mystep = MyStep(...)`

and then you can call

`step = mystep(D; kwargs...)`

to compute the step size.

`AbstractStep`s should be able to construct themselves without arguments:
`MyStep()`, ideally with default parameters if necessary.
Or they can store a single function that calculates the step (like `LipschitzStep`).

Typical keywords at call time are:
`n`: Which factor the step is for
`objective`: The objective function (takes the entire iterate as input)
`gradient`: How to calculate the gradient of the objective w.r.t. the `n`th factor (takes the entire iterate as input)
`x_last`: The previous iterate
`current_gradient`: precomputed `gradient(x)`

for example
`U.step(x; n, x_last, gradient, current_gradient=grad, objective, kwargs...)`
"""
abstract type AbstractStep <: Function end

"""
    LipschitzStep <: AbstractStep

Has a single property `lipschitz` which stores a function for calculating the Lipschitz
constant of the gradient with respect to a factor.
"""
struct LipschitzStep <: AbstractStep
    lipschitz::Function
end

"""
    (step::LipschitzStep)(x; kwargs...)

Computes the step size 1/L.
"""
function (step::LipschitzStep)(x; kwargs...)
    L = step.lipschitz(x)
    try
        return L^(-1)  # allow for Lipschitz to be a diagonal matrix
    catch
        @warn "Could not invert the Lipschitz constant to get a stepsize. Ignoring zero coordinates."

        return _safe_invert.(L)
    end
end

_safe_invert(x) = iszero(x) ? x : x^(-1)

function (step::LipschitzStep)(x::Tucker; kwargs...)
    L = step.lipschitz(x)
    if typeof(L) <: Tuple # Currently the only case is when we are updating the core of a Tucker factorization
                          # Using this condition as a way to tell if it is the core we are calculating the constant for
        return map(X -> X^(-1), L)
    else
        return L^(-1) # allow for Lipschitz to be a diagonal matrix
    end
end
#LipschitzStep(L::Real) = 1/L

struct ConstantStep <: AbstractStep
    stepsize::Float64
end

(step::ConstantStep)(x; kwargs...) = step.stepsize

struct SPGStep <: AbstractStep
    min::Float64
    max::Float64
    function SPGStep(min, max)
        min < max || throw(ArgumentError("min $min must be less than max $max"))
        new(min, max)
    end
end

SPGStep(; min=1e-10, max=1e10) = SPGStep(min, max)

# option to override the set defaults from step
# TODO SPG has a linesearch/negative momentum update part to the fill iteration
# but in the best case, this linesearch just uses the value given by this step
# so I will skip implementing it for now, but may want to add that once
# I add a line search
function (step::SPGStep)(x; n, x_last, gradient, current_gradient, stepmin=step.min, stepmax=step.max, kwargs...)
    # Extract current and last iterates and gradients
    grad = current_gradient
    grad_last = gradient(x_last; kwargs...)
    x = factor(x, n)
    x_last = factor(x_last, n)

    # Compute the step
    s = x - x_last
    y = grad - grad_last
    sy = (s ⋅ y)
    if sy <=0 #TODO check why (s ⋅ y) < 0 means we should take stepmax and not stepmin
        return stepmax
    else
        suggested_step = (s ⋅ s) / sy
        return clamp(suggested_step, stepmin, stepmax) # safeguards to ensure step is within reasonable bounds
    end
end

"""
    SecantStep <: AbstractStep

Approximates the local smoothness (Lipschitz constant of the gradient) using finite differences.

The step is norm(x - x_last) / norm(grad - grad_last).

"""
struct SecantStep <: AbstractStep end

function (step::SecantStep)(x; n, x_last, gradient, current_gradient, kwargs...)
    # Extract current and last iterates and gradients
    grad = current_gradient
    grad_last = gradient(x_last; kwargs...)
    x = factor(x, n)
    x_last = factor(x_last, n)  

    return norm(x - x_last) / norm(grad - grad_last) # always the Euclidean norm (induced by the inner product/operation of gradient ⋅ vector)
end

"""
    ArmijoStep <: AbstractStep 

Armijo steplength rule selects the largest step t=β^p such that

`f(x) - f(x - t∇f(x)) ≥ δt‖∇f(x)‖^2`

where `β` and `δ` are between `(0, 1)`.

This ensures the new point `x_new = x - t∇f(x)` reduces the objective by a set amount:

`f(x_new) ≤ f(x) - δt‖∇f(x)‖^2`.

Parameters `β` and `δ` default to `0.5`.
"""
struct ArmijoStep <: AbstractStep 
    β::Float64
    δ::Float64
    function ArmijoStep(; β=0.5, δ=0.5)
        0 ≤ β ≤ 1 || throw(ArgumentError("β must be between 0 and 1, got $β"))
        0 ≤ δ ≤ 1 || throw(ArgumentError("δ must be between 0 and 1, got $δ"))
        new(β, δ)
    end
end

function (step::ArmijoStep)(X; n, objective, current_gradient, kwargs...)
    X_new = copy(X)
    x_new = factor(X_new, n)
    x = factor(X, n)

    β = step.β
    δ = step.δ
    t = 1

    current_objective = objective(X)
    g = current_gradient
    g_norm = norm2(g)

    @. x_new = x - t * g
    threshold = δ*g_norm

    while current_objective - objective(X_new) < t*threshold
        t *= β
        @. x_new = x - t * g
    end

    return t # slight inefficiency where we don't also return the new iterate `x_new`
end