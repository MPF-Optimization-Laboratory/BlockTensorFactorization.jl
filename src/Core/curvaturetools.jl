"""Short helpers and operations related to finite differences and curvature"""

"""
    d_dx(y::AbstractVector{<:Real})

Approximate first derivative with finite elements.

``\\frac{d}{dx}y[i] \\approx \\frac{1}{2\\Delta x}(y[i+1] - y[i-1])``

Assumes `y[i] = y(x[i])` are samples with unit spaced inputs `Δx = x[i+1] - x[i] = 1`.
"""
function d_dx(y::AbstractVector{<:Real})
    if length(y) < 3
        throw(ArgumentError("y must have length at least 3, got $(length(y))"))
    end

    d = similar(y)
    each_i = eachindex(y)

    # centred estimate
    for i in each_i[begin+1:end-1]
        d[i] = (-y[i-1] + y[i+1])/2
    end

    # three point forward/backward estimate
    i = each_i[begin+1]
    d[begin] = (-3*y[i-1] + 4*y[i] - y[i+1])/2

    i = each_i[end-1]
    d[end] = (y[i-1] - 4*y[i] + 3*y[i+1])/2
    return d
end

"""
    d2_dx2(y::AbstractVector{<:Real})

Approximate second derivative with finite elements.

``\\frac{d^2}{dx^2}y[i] \\approx \\frac{1}{\\Delta x^2}(y[i-1] - 2y[i] + y[i+1])``

Assumes `y[i] = y(x[i])` are samples with unit spaced inputs `Δx = x[i+1] - x[i] = 1`.
"""
function d2_dx2(y::AbstractVector{<:Real})
    if length(y) < 3
        throw(ArgumentError("y must have length at least 3, got $(length(y))"))
    end

    d = similar(y)
    for i in eachindex(y)[begin+1:end-1]
        d[i] = y[i-1] - 2*y[i] + y[i+1]
    end
    # Assume the same second derivative at the end points
    d[begin] = d[begin+1]
    d[end] = d[end-1]
    return d
end

"""
    cubic_spline_coefficients(y::AbstractVector{<:Real}; h=1)

Calculates the list of coefficients `a`, `b`, `c`, `d` for an interpolating spline of `y[i]=y(x[i])`.

The spline is defined as ``f(x) = g_i(x)`` on ``x[i] \\leq x \\leq x[i+1]`` where

``g_i(x) = a[i](x-x[i])^3 + b[i](x-x[i])^2 + c[i](x-x[i]) + d[i]``

Uses the following boundary conditions
- ``g_1(x[1]-h) = 1`` (i.e. the ``y``-intercept is ``(0,1)`` for uniform spaced `x=1:I`)
- ``g_I(x[I]+h) = y[I]`` (i.e. repeated right end-point)
- ``g_I''(x[I]+h) = 0`` (i.e. flat/no-curvature one spacing after end-point)
"""
function cubic_spline_coefficients(y::AbstractVector{<:Real}; h=1)
    # Set up variables
    n = length(y)
    #T = eltype(y)
    f = diff([y; y[end]]) # use diff([y; zero(T)]) to clamp at a y value of 0 instead of a repeated boundary condition

    # solve the system Mb=v
    M = spline_mat(n)
    v = 3/h^2 .* [1 - 2y[1] + y[2]; diff(f); 0]
    b = M \ v

    # use b to find the other coefficients
    c = [f[i]/h - h/3*(b[i+1] + 2b[i]) for i in 1:n]
    a = diff(b) ./ 3h
    d = copy(y)

    # truncate b from length n+1 to n
    return a, b[1:end-1], c, d
end

"""
    spline_mat(n)

Creates the `Tridiagonal` matrix to solve for coefficients `b`. See [`cubic_spline_coefficients`](@ref).
"""
function spline_mat(n)
    du = [0; ones(Int, n-1)]
    dd = [6; 4*ones(Int, n-1) ; 1]
    dl = [ones(Int, n-1); 0]

    return Tridiagonal(dl, dd, du)
end

"""
    make_spline(y::AbstractVector{<:Real}; h=1)

Returns a function f(x) that is an interpolating/extrapolating spline for y, with
uniform stepsize h between the x-values of the knots.
"""
function make_spline(y::AbstractVector{<:Real}; h=1)
    a, b, c, d = cubic_spline_coefficients(y::AbstractVector{<:Real}; h)
    n = length(y)

    function f(x)
        i = Int(floor(x))

        # find which spline piece to use
        # extrapolating from the first or last spline if needed
        if i < 1
            i = 1
        elseif i > n
            i = n
        end

        h = x - i

        return a[i]*h^3 + b[i]*h^2 + c[i]*h + d[i]
    end

    return f
end

"""
    d_dx_and_d2_dx2_spline(y::AbstractVector{<:Real}; h=1)

Extracts the first and second derivatives of the splines from y at the knots
"""
function d_dx_and_d2_dx2_spline(y::AbstractVector{<:Real}; h=1)
    _, b, c, _ = cubic_spline_coefficients(y::AbstractVector{<:Real}; h)
    dy_dx = c
    dy2_dx2 = 2b
    return dy_dx, dy2_dx2
end


"""
    curvature(y::AbstractVector{<:Real}; method=:finite_differences)

Approximates the signed curvature of a function given evenly spaced samples.

# Possible `method`s
- `:finite_differences`: Approximates first and second derivative with 3rd order finite differences. See [`d_dx`](@ref) and [`d2_dx2`](@ref).
- `:splines`: Curvature of a third order spline. See [`d_dx_and_d2_dx2_spline`](@ref).
- `:circles`: Inverse radius of a circle through rolling three points. See [`circle_curvature`](@ref).
- `:breakpoints`: WARNING does not compute a value that approximates the curvature of a continuous function. Computes the inverse least-squares error of `f.(eachindex(y); z)` and `y` for all `z in eachindex(y)` where `f(x; z) = a + b(min(x, z) - z) + c(max(x, z) - z)`. Useful if `y` looks like two lines. See [`breakpoint_curvature`](@ref).
"""
function curvature(y::AbstractVector{<:Real}; method=:finite_differences, kwargs...)
    if method == :finite_differences
        dy_dx = d_dx(y; kwargs...)
        dy2_dx2 = d2_dx2(y; kwargs...)
        return @. dy2_dx2 / (1 + dy_dx^2)^1.5
    elseif method == :splines
        dy_dx, dy2_dx2 = d_dx_and_d2_dx2_spline(y; h=1)
        return @. dy2_dx2 / (1 + dy_dx^2)^1.5
    elseif method == :circles
        return circle_curvature(y; h=1)
    elseif method == :breakpoints
        return breakpoint_curvature(y)
    else
        throw(ArgumentError("method $method not implemented"))
    end
end

"""
    standard_curvature(y::AbstractVector{<:Real}; method=:finite_differences)

Approximates the signed curvature of a function, scaled to the unit box ``[0,1]^2``.

Assumes the function is 1 at 0 and (after x dimension is scaled) 0 at 1.

See [`curvature`](@ref).


# Possible `method`s
- `:finite_differences`: Approximates first and second derivative with 3rd order finite differences. See [`d_dx`](@ref) and [`d2_dx2`](@ref).
- `:splines`: Curvature of a third order spline. See [`d_dx_and_d2_dx2_spline`](@ref).
- `:circles`: Inverse radius of a circle through rolling three points. See [`circle_curvature`](@ref).
- `:breakpoints`: WARNING does not compute a value that approximates the curvature of a continuous function. Computes the inverse least-squares error of `f.(eachindex(y); z)` and `y` for all `z in eachindex(y)` where `f(x; z) = a + b(min(x, z) - z) + c(max(x, z) - z)`. Useful if `y` looks like two lines. See [`breakpoint_curvature`](@ref).
"""
function standard_curvature(y::AbstractVector{<:Real}; method=:finite_differences, kwargs...)
    Δx = 1/length(y)
    # An interval 0:10 has length(0:10) = 11, but measure 10-0 = 10 so we may think to use 1/(length(y) - 1), but we need to consider the left end point of y is at 1/length(y) and not zero.
    if method == :finite_differences
        y = [1; y; 0]
        y_max = maximum(y)
        dy_dx = d_dx(y; kwargs...) / (Δx * y_max)
        dy2_dx2 = d2_dx2(y; kwargs...) / (Δx^2 * y_max)
        curvature =  @. dy2_dx2 / (1 + dy_dx^2)^1.5
        return curvature[begin+1:end-1]
    elseif method == :splines
        # y_max = 1
        dy_dx, dy2_dx2 = d_dx_and_d2_dx2_spline(y; h=Δx)
        return @. dy2_dx2 / (1 + dy_dx^2)^1.5
    elseif method == :circles
        return circle_curvature(y / max(1,maximum(y)); h=Δx)
    elseif method == :breakpoints
        return breakpoint_curvature(y) # best breakpoint unaffected by scaling and stretching
    else
        throw(ArgumentError("method $method not implemented"))
    end
end

"""
    circle_curvature(y::AbstractVector{<:Real}; h=1, estimate_endpoints=true)

Inverse radius of a the circle passing through each 3 adjacent points on `y`,
`(0,y[i-1])`, `(h,y[i])`, and `(2h,y[i+1])`.

If `estimate_endpoints=true`, assumes the function that y comes from is 1 to the left of the given values, and 0 to the right. This is typical of relative error decay as a function of rank.
If `false`, pads the boundary with the adjacent curvature.

See [`three_point_circle`](@ref).
"""
function circle_curvature(y::AbstractVector{<:Real}; h=1, estimate_endpoints=true)
    k = zero(y)
    a, b, c = 0, h, 2h
    eachindex_k = eachindex(k)
    for i in eachindex_k[2:end-1]
        k[i] = signed_circle_curvature((a,y[i-1]),(b,y[i]),(c,y[i+1]))
    end

    if estimate_endpoints
        i = eachindex_k[1]
        k[i] = signed_circle_curvature((a, 1),(b,y[i]),(c,y[i+1]))
        i = eachindex_k[end]
        k[i] = signed_circle_curvature((a,y[i-1]),(b,y[i]),(c, 0))
    else
        k[1] = k[2]
        k[end] = k[end-1]
    end

    return k
end

"""
    three_point_circle((a,f),(b,g),(c,h))

Calculates radius `r` and center point `(p, q)` of the circle passing through the three points
in the xy-plane.

# Example
```julia
r, (p, q) = three_point_circle((1,2), (2,1), (5,2))
(r, (p, q)) == (√5, (3, 3))
```
"""
function three_point_circle((a,f),(b,g),(c,h))
    fg = f-g
    gh = g-h
    hf = h-f
    ab = a-b
    bc = b-c
    ca = c-a
    a2 = a^2
    b2 = b^2
    c2 = c^2
    f2 = f^2
    g2 = g^2
    h2 = h^2
    p = (a2*gh + b2*hf + c2*fg - gh*hf*fg) / (a*gh + b*hf + c*fg) / 2
    q = (f2*bc + g2*ca + h2*ab - bc*ca*ab) / (f*bc + g*ca + h*ab) / 2
    r = sqrt((a-p)^2 + (f-q)^2)
    return r, (p, q)
end

"""
    signed_circle_curvature((a,f),(b,g),(c,h))

Signed inverse radius of the circle passing through the 3 points in the xy-plane.

See [`three_point_circle`](@ref).
"""
function signed_circle_curvature((a,f),(b,g),(c,h))
    @assert a < b < c
    r, _ = three_point_circle((a,f),(b,g),(c,h))
    sign = g > (f+h)/2 ? -1 : 1
    return sign / r
end

"""
    breakpoint_model_coefficients(xs, ys, breakpoint)

Least squares fit data ``(x_i, y_i)``

``\\min_{a,b,c} 0.5\\sum_{i} (f(x_i; a,b,c) - y_i)^2``

with the model

``f(x; a,b,c) = a + b(\\min(x, z) - x) + c(\\max(x, z) - x)``

for some fixed ``z``.
"""
function breakpoint_model_coefficients(xs, ys, z)
    n = length(xs)
    @assert n == length(ys)
    M = hcat(ones(n), (min.(xs, z) .- z), (max.(xs, z) .- z))
    a, b, c = M \ ys
    return a, b, c
end
# TODO add an option to compute a cheaper but less accurate model
# a = z; b = (1 - y[z]) / z ; c = y[z] / (R_max - z)
# This fixes the model to the three points (0,1), (z, y[z]), and (R_max, 0)

"""
    breakpoint_model(a, b, c, z)

Returns a function `x -> a + b*(min(x, z) - z) + c*(max(x, z) - z)`.
"""
breakpoint_model(a, b, c, z) = x -> a + b*(min(x, z) - z) + c*(max(x, z) - z)

"""
    breakpoint_error(xs, ys, z)

Squared L2 error between the best breakpoint model (with breakpoint `z`) evaluated at `xs` and the data `ys`.

See [`breakpoint_model`](@ref) and [`breakpoint_model_coefficients`](@ref).
"""
function breakpoint_error(xs, ys, z)
    a, b, c = breakpoint_model_coefficients(xs, ys, z)
    f = breakpoint_model(a, b, c, z)
    return norm2(@. f(xs) - ys)
    # equivalent to sum(((x, y),) -> (f(x) - y)^2, zip(xs, ys))
end

"""
    best_breakpoint(xs, ys; breakpoints=xs)

Breakpoint `z in breakpoints` that minimizes the [`breakpoint_error`](@ref).
"""
best_breakpoint(xs, ys; breakpoints=xs) = argmin(z -> breakpoint_error(xs, ys, z), breakpoints)

"""
    breakpoint_curvature(y)

This is a hacked way to fit the data `y` with a breakpoint model,
which can be called by `k = standard_curvature(...; model=:breakpoints)`

This lets us call `argmax(k)` to get the breakpoint that minimizes the model error.

See [`breakpoint_model_coefficients`](@ref).
"""
function breakpoint_curvature(y)
    x = eachindex(y)
    errors = [breakpoint_error(x, y, z) for z in x]
    return 1 ./ errors
end
