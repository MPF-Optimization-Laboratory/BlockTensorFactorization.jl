# Rank Estimation

## Overview

In some applications, we may have a tenor we believe to be closely approximated by a low rank tenor, but do not know what rank to use. In this setting, use `rank_detect_factorize` rather than `factorize` without a `rank=R` keyword. The estimated rank will be added to `kwargs`, and the `decomposition` at this rank returned.

```julia
R = 3
T = Tucker1((10, 10, 10), R)
Y = array(T)
options = (model=Tucker1, curvature_method=splines) # default values
decomposition, stats, kwargs, final_rel_errors = rank_detect_factorize(Y; options...)

@assert kwargs[:rank] == R
```

The rank will be estimated by factorizing the input tensor at every possible rank from 1 up to the maximum rank and recording the relative error. We can pick the rank which balances small error (good fit) and small rank (low complexity). Our criteria is the rank which maximizes the curvature of the error vs. rank function, or in the case of the `:breakpoints` method, a proxy for the curvature.

Note `rank_detect_factorize` also returns the stats and final relative errors of each rank tested.

```@docs; canonical=false
rank_detect_factorize
max_possible_rank
```

!!! tip
    If you run `rank_detect_factorize` with different methods and get the same rank, you can feel more confident in the detected rank.


## Estimating Curvature

This package has implemented a few methods for estimating curvature. These are the following list of methods.

```julia
:finite_differences
:circles
:splines
```

We can also use the following curvature-proxy methods.

```julia
:breakpoints
```

They can be passed to `curvature` and `standard_curvature` as a `method` keyword, or `rank_detect_factorize` with the `curvature_method` keyword.

```@docs; canonical=false
curvature
standard_curvature
```

These methods have been specialized to typical relative error vs. rank curves. We assume a relative error of ``1`` at rank ``0`` since the only rank ``0`` tenor is the ``0`` tenor. And that the relative error at the maximum possible rank is ``0``, and stays zero if we try to use a larger rank.

## How do these methods work?

### Finite Differences

We approximate the first and second derivatives separately with three point finite differences and calculate the curvature with the formula

```math
k(x) = \frac{y''(x)}{(1 + y'(x)^2)^{3/2}}.
```

The derivatives are approximated using centred three point finite differences,[^1]

[^1]: M. Abramowitz and I. A. Stegun, "Handbook of mathematical functions: with formulas, graphs and mathematical tables", Unabridged, Unaltered and corr. Republ. of the 1964 ed. in Dover books on advanced mathematics. New York: Dover publ, 1972. (Table 25.2, p. 914)

```math
y'(x_i) \approx \frac{1}{2\Delta x}(y_{i+1} - y_{i-1})\quad\text{and}\quad y''(x_i)\approx \frac{1}{\Delta x^2}(y_{i+1} - 2y_i + y_{i-1}).
```

The end-points use forward/back-differences. For the first derivative, this is

```math
y'(x_1) \approx \frac{1}{2\Delta x}(-3y_{1} + 4y_2 - y_{3})\quad\text{and}\quad y'(x_I) \approx \frac{1}{2\Delta x}(-y_{I-2} + 4y_{I-1} - 3y_{I}),
```

and for the second derivative, this is

```math
y''(x_1) \approx \frac{1}{\Delta x^2}(y_{1} - 2y_2 + y_{3})\quad\text{and}\quad y''(x_I) \approx \frac{1}{\Delta x^2}(y_{I-2} - 2y_{I-1} + y_{I}).
```

```@docs; canonical=false
BlockTensorFactorization.Core.d_dx
BlockTensorFactorization.Core.d2_dx2
```


### Splines

We calculate a third order [spline](https://en.wikipedia.org/wiki/Spline_(mathematics))

```math
g_i(x) = a_i(x-x_i)^3+b_i(x-x_i)^2+c_i(x-x_i)+d_i,\quad x_1 \leq x \leq x_{i+1}
```

and use the coefficients to calculate the curvature of the spline

```math
k(x_i) = \frac{2b_i}{(1+c_i^2)^{3/2}}.
```

We assume the following boundary conditions;
1) ``g_{1}(x_{1}-\Delta x)=1``, ``y``-intercept ``(0,1)``
2) ``g_{I}(x_{I}+\Delta x)=y_{I}``, repeated right end-point (``y_{I+1}=y_{I}``)
3) ``g_{I}''(x_{I}+\Delta x)=0``, flat right end-point,

in addition to the usual continuity and smoothness assumptions of a third order spline:
1) ``g_{i}(x_{i})=y_{i}``
2) ``g_{i}(x_{i+1})=g_{i+1}(x_{i+1})``
3) ``g_{i}'(x_{i+1})=g_{i+1}'(x_{i+1})``
4) ``g_{i}''(x_{i+1})=g_{i+1}''(x_{i+1})``.

We can solve for the coefficients by first solving the tri-diagonal system ``Mb=v``, where

```math
M=\begin{bmatrix}
6 & 0 &  & & \\
1 & 4 & 1 & & \\
& \ddots& \ddots& \ddots& \\
& & 1 & 4 & 1& \\
 & &  & 0 & 1
\end{bmatrix},\quad
b =\begin{bmatrix}
b_{1}  \\
b_{2}\\
\vdots \\
b_{I} \\
b_{I+1}
\end{bmatrix},\quad \text{and} \quad
v = \frac{3}{\Delta x^2}\begin{bmatrix}
1-2y_{1}+y_{2} \\
y_{3}-y_{1} \\
\vdots \\
y_{I+1}-y_{I-1} \\
0
\end{bmatrix},
```

and then calculate

```math
a_{i} = \frac{1}{3\Delta x}(b_{i+1}-b_{i})\quad \text{and}\quad c_{i}=\frac{{y_{i+1}-y_{i}}}{\Delta x}-\frac{\Delta x}{3}(b_{i+1}-2b_{i}).
```

```@docs
BlockTensorFactorization.Core.cubic_spline_coefficients
BlockTensorFactorization.Core.spline_mat
BlockTensorFactorization.Core.make_spline
BlockTensorFactorization.Core.d_dx_and_d2_dx2_spline
```

### Circles

We compute the radius ``r_i`` of a circle passing through three neighbouring points ``(x_{i-1}, y_{i-1})``, ``(x_{i}, y_{i})``, and ``(x_{i+1}, y_{i+1})``. The curvature magnitude is approximately the inverse of this radius;

```math
\lvert k(x_i)\rvert \approx \frac{1}{r_i}.
```

We assume two additional points so we can estimate the curvature at the boundary. They are

```math
(x_{0}, y_{0}) = (0, 1)\quad\text{and}\quad (x_{I+1}, y_{I+1}) = (x_{I}+\Delta x, 0).
```

To obtain a the signed curvature ``k(x_i)``, we check if the middle point ``(x_{i}, y_{i})`` is above (negative curvature) or below (positive curvature) the line segment connecting ``(x_{i-1}, y_{i-1})`` and ``(x_{i+1}, y_{i+1})``.

```@docs
BlockTensorFactorization.Core.three_point_circle
BlockTensorFactorization.Core.circle_curvature
BlockTensorFactorization.Core.signed_circle_curvature
```


### Two-line Breakpoint

This method does **not** approximate the curvature of a function passing through the points ``\{(x_i, y_i)\}``, but picks the best rank based on the optimal breakpoint. The optimal breakpoint ``r`` is the breakpoint of segmented linear model that minimizes the model error.[^2] This means

[^2]: J. E. Saylor, K. E. Sundell, and G. R. Sharman, "Characterizing sediment sources by non-negative matrix factorization of detrital geochronological data," Earth and Planetary Science Letters, vol. 512, pp. 46–58, Apr. 2019, doi: [10.1016/j.epsl.2019.01.044](https://doi.org/10.1016/j.epsl.2019.01.044). (Section 3.2)

```math
r = \argmin_{z \in \{1,\dots, R\}} \sum_{i=1}^I (f_z(x_i) - y_i)^2,
```

where

```math
f_z(x; a_z, b_z, c_z) = a_z + b_z(\min(x,z) - z) + c_z(\max(x,z) - z)
```

and the coefficients ``(a_z, b_z, c_z)`` are selected to minimize the error between the model and the data,

```math
(a_z, b_z, c_z) = \argmin_{a,b,c\in\mathbb{R}} \sum_{i=1}^I (f_z(x_i;a,b,c) - y_i)^2.
```

These coefficients have the closed form solution

```math
\begin{bmatrix}
a_{z} \\
b_{z} \\
c_{z}
\end{bmatrix} =(M^\top M)^{-1}M^\top \begin{bmatrix}
y_{1} \\
\vdots \\
y_{I}
\end{bmatrix},\quad\text{where}\quad M = \begin{bmatrix}
1 & \min(x_{1},z) - z&\max(x_{1},z) - z \\
\vdots & \vdots & \vdots\\
1 & \min(x_{I},z) - z&\max(x_{I},z) - z
\end{bmatrix}.
```

Geometrically, the breakpoint model is two half-infinite lines on ``(-\infty, z]`` and ``[z,\infty)`` that meet continuously at ``(z,a)`` with slopes ``b`` and ``c`` respectively.

```@docs
BlockTensorFactorization.Core.best_breakpoint
BlockTensorFactorization.Core.breakpoint_error
BlockTensorFactorization.Core.breakpoint_model
BlockTensorFactorization.Core.breakpoint_model_coefficients
BlockTensorFactorization.Core.breakpoint_curvature
```
