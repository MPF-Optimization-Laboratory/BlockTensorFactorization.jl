# Multiscale Factorization

## Quick Start

In some applications, the input tensor $Y$ to be factorized is a discretization of continuous functions. For example,

$$Y[i,j,k] = f_i(x_j, y_k)$$

for continuous functions $f_i$ and $(i,j,k)\in[I]\times[J]\times[K]$. Instead of calling `factorize`

```julia
factorize(Y; kwargs..)
```

we can use `multiscale_factorize`, and pass along information about which dimension come from continuously varying values.

```julia
multiscale_factorize(Y; continuous_dims=(2,3), kwargs...)
```

For fine discretizations, this is often faster and less memory intensive.

## Motivating example

Say we have an order $3$-tensor $Y$ with entries

$$Y[i, j, k] = x[i] y[j] z[k]$$

for $(i,j,k) \in [65]\times[129]\times[257]$ representing a discretization of the box $[-1, 1]\times [0, 10]\times [0,1]$.

We might construct this in Julia with the following code.

```julia
f(x, y, z) = x*y*z
x = reshape(range(-1, 1, length=65), 65, 1, 1)
y = reshape(range(0, 10, length=129), 1, 129, 1)
z = reshape(range(0, 1, length=257), 1, 1, 257)
Y = f.(x, y, z)
```

It is true that `Y` is a CP-rank-$1$ tensor. So we could recover the factors `x`, `y`, and `z` with

```julia
decomposition, stats, kwargs = multiscale_factorize(Y;
    model=CPDecomposition, rank=1, continuous_dims=(1,2,3))
x, y, z = factors(decomposition)
```

The grid $(i,j,k) \in [65]\times[129]\times[257]$ we chose to discretize the function $f(x,y,z)=xyz$ was arbitrary and we could just as easily discretized $f$ on a smaller grid $(i,j,k)\in[10]^3$ or larger one $(i,j,k)\in[257]^3$.


In this setting, we can first decompose a coarse version of `Y`, for example only using every second grid-points in each dimension

```julia
Y_coarse = Y[begin:2:end, begin:2:end, begin:2:end]
```

and compute a coarse decomposition

```julia
model = CPDecomposition
rank = 1
decomposition_coarse, _, _ = factorize(Y_coarse; model, rank)
x_coarse, y_coarse, z_coarse = factors(decomposition_coarse)
```

We could then initialize the fine scale factorization using an interpolation of the coarse factors found.

```julia
init = interpolate.(factors(decomposition))
factorize(Y; model, rank, init)
```

We can actually do this starting from the most coarse scale with only $3$ points, and gradually refine the decomposition over multiple scale. This is what `multiscale_factorize` does.

```@docs; canonical=false
multiscale_factorize
```

## Related exported functions

```@docs; canonical=false
coarsen
interpolate
scale_constraint
```
