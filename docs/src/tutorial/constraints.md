# Constrained Factorization

## Example
A number of constraints are ready-to-use out of the box. Say you would like to perform a rank 5 CP decomposition of a 3rd order tensor, such that every column is constrainted to the simplex. This can be accomplished by the following code.

```julia
X, stats, kwargs = factorize(Y; model=CPDecomposition, rank=5, constraints=simplex_cols!)
```

Only one constraint is given when there are 3 factors in the CP decomposition of Y, so it is assumed this constraint applies to every factor.

Say you instead you only want the 1st factor to have columns constrained to the simplex, and the 3rd factor to be nonnegative. You can use the following code.

```julia
X, stats, kwargs = factorize(Y; model=CPDecomposition, rank=5,
    constraints=[simplex_cols!, noconstraint, nonnegative!])
```

## Ready-to-use Constraints

### Entrywise
```@docs; canonical=false
Entrywise
```

The constraints `nonnegative!` and `binary!` ensure each entry in a factor is nonnegative and either 0 or 1 respectively.

```@docs; canonical=false
nonnegative!
binary!
```

If you want every entry to be in the closed interval `a`, `b`, you can use `IntervalConstraint(a, b)`.

```@docs; canonical=false
IntervalConstraint
```

### Normalizations

```@docs; canonical=false
ProjectedNormalization
```

These ensure a factor is normalized according to the L1, L2, & L infinity norms. This is accomplished through a Euclidean projections onto the unit ball.

```@docs; canonical=false
l1normalize!
l2normalize!
linftynormalize!
```

Each come in versions that constrain each row, column, order-1 slice, & order-(1, 2) slice to the associated unit norm ball.

```
l1normalize_rows!
l1normalize_cols!
l1normalize_1slices!
l1normalize_12slices!

l2normalize_rows!
l2normalize_cols!
l2normalize_1slices!
l2normalize_12slices!

linftynormalize_rows!
linftynormalize_cols!
linftynormalize_1slices!
linftynormalize_12slices!
```

!!! warning "Warning"
    Projection onto the unit norm ball from the origin is not unique at the origin so do not expect consistent behaviour with an all-zeros input.

### Scaling Constraints
```@docs; canonical=false
ScaledNormalization
```

Each previously listed normalization has an associated scaled normalization. These ensure the relevant subarrays are normalized, but rather than enforce these by a Euclidean projection, they simply divide by its norm. This is equivalent to the Euclidean projection onto the L2 norm ball, but is a different operation for the L1 and L infinity balls. This offers the advantage that other factors can be "rescaled" to compensate for this division which is not normally possible with the projections onto the L1 and L infinity balls.

!!! note "Note"
    By default, when these constraints are applied, they will "rescale" the other factors to minimize the change in the product of all the factors.

!!! details "Example"
    Say we are performing CPDecomposition on a matrix `Y`. This is equivalent to factorizing `Y = A * B'`. If we would like all columns of `B` (rows of `B` transpose) to be on the L1 ball, rather than projecting each column, we can instead divide each column of `B` by its L1 norm, and multiply the associated column of `A` by this amount. This has the advantage of enforcing our constraint without effecting the product `A * B'`, whereas a projection would possibly change this product.

```
l1scale!
l1scale_rows!
l1scale_cols!
l1scale_1slices!
l1scale_12slices!

l2scale!
l2scale_rows!
l2scale_cols!
l2scale_1slices!
l2scale_12slices!

linftyscale!
linftyscale_rows!
linftyscale_cols!
linftyscale_1slices!
linftyscale_12slices!
```

There is also a set of constraints that ensure the order-(1,2) slices are scaled on average. This makes preserving a Tucker1 product possible where you would like each order-(1,2) normalized.

```
l1scale_average12slices!
l2scale_average12slices!
linftyscale_average12slices!
```

### Simplex Constraint
Similar to the L1 normalization constraint, these constraints ensure the relevant subarrays are on the L1 ball. But these also ensure all entries are positive. This is enforced with a single Euclidean projection onto the relevant simplex.

```
simplex!
simplex_cols!
simplex_1slices!
simplex_12slices!
```

### List of built-in `ProjectedNormalization`s and `ScaledNormalization`s

```@docs
BUILT_IN_CONSTRAINTS
```

```
l1normalize!
l1normalize_rows!
l1normalize_cols!
l1normalize_1slices!
l1normalize_12slices!
l2normalize!
l2normalize_rows!
l2normalize_cols!
l2normalize_1slices!
l2normalize_12slices!
linftynormalize!
linftynormalize_rows!
linftynormalize_cols!
linftynormalize_1slices!
linftynormalize_12slices!
simplex!
simplex_rows!
simplex_cols!
simplex_1slices!
simplex_12slices!
l1scale!
l1scale_rows!
l1scale_cols!
l1scale_1slices!
l1scale_12slices!
l1scale_average12slices!
l2scale!
l2scale_rows!
l2scale_cols!
l2scale_1slices!
l2scale_12slices!
l2scale_average12slices!
linftyscale!
linftyscale_rows!
linftyscale_cols!
linftyscale_1slices!
linftyscale_12slices!
linftyscale_average12slices!
```

<!--
```@example
using BlockTensorFactorization # hide
println.(BUILT_IN_CONSTRAINTS);
```
-->

### Linear Constraints

```@docs; canonical=false
LinearConstraint
```

## Advanced Constraints

### Ending with a different set of constraints

It is possible to apply a different set of constraints at the end of the algorithm than what is enforced during the iterations. For example, you can perform nonnegative Tucker factorization of a tensor `Y`, but apply a simplex constraint on the core at the very end.

```julia
X, stats, kwargs = factorize(Y; model=Tucker, rank=2,
    constraints=nonnegative!,
    final_constraints=[simplex!, noconstraint, noconstraint, noconstraint])
```

In the case where constraints effect other factors (e.g. `l2scale!`), you may want to perform a final pass of the constraints to ensure each factor is scaled correctly, without rescaling/effecting other factors.

```julia
X, stats, kwargs = factorize(Y; model=CPDecomposition, rank=3,
    constraints=l2scale_cols!,
    constrain_output=true)
```

### Composing Constraints

`AbstractConstraint` types can be composed with `\circ` (and hitting tab to make `∘`) creating a `ComposedConstraint`.

```@docs; canonical=false
ComposedConstraint
```

!!! warning
    All `ComposedConstraint`s do is apply the two constraints in series and does *not* do anything intelligent like finding the intersection of the constraints. For this reason, the following three constraints are all different.

    ```julia
    l1normalize! ∘ nonnegative!
    nonnegative! ∘ l1normalize!
    simplex!
    ```

### Custom Constraints
You can define your own `ProjectedNormalization`, `ScaledNormalization`, or `Entrywise` constraint using the following constructors.

```julia
ScaledNormalization(norm; whats_normalized=identityslice, scale=1)
ProjectedNormalization(norm, projection; whats_normalized=identityslice)
Entrywise(apply, check)
```

In fact, these are how the ready-to-use constraints are made. Here are some examples.

```julia
l2scale_1slices! = ScaledNormalization(l2norm; whats_normalized=(x -> eachslice(x; dims=1)))
l1normalize_rows! = ProjectedNormalization(l1norm, l1project!; whats_normalized=eachrow)
nonnegative! = Entrywise(ReLU, isnonnegative)
IntervalConstraint(a, b) = Entrywise(x -> clamp(x, a, b), x -> a <= x <= b)
```

You can also make a custom constraint with `GenericConstraint`.

```@docs; canonical=false
GenericConstraint
```

## Manual Constraint Updates

You can manually define the [`ConstraintUpdate`](@ref) that gets applied as part of the block decomposition method. These will be automatically inserted into the order of updates immediately following the last update of the matching block with [`smart_interlace!`](@ref).

As an example, if we are performing CP decomposition on an order 3 tensor, the unconstrained block optimization would look something like this.

```julia
BlockUpdate(
    GradientDescent(1, gradient, step)
    GradientDescent(2, gradient, step)
    GradientDescent(3, gradient, step)
)
```

Here the 1, 2, and 3 denote which factor gets updated. If we want to apply a simplex constraint to the second factor, and nonnegative to the 3rd, you can do the following.

```julia
X, stats, kwargs = factorize(Y; model=CPDecomposition, rank=5, constraints=[ConstraintUpdate(3, nonnegative!), Projection(2, simplex!)])
```

This would result in the following block update.

```julia
BlockUpdate(
    GradientDescent(1, gradient, step)
    GradientDescent(2, gradient, step)
    Projection(2, simplex!)
    GradientDescent(3, gradient, step)
    Projection(3, nonnegative!)
)
```

Note the order `[ConstraintUpdate(3, nonnegative!), Projection(2, simplex!)]` does *not* matter. Also, using the (abstract) constructor `ConstraintUpdate` will reduce to the concrete types when possible.

```@docs; canonical=false
ConstraintUpdate
```

When using `ScaledNormalization`s, you may want to manually define what gets rescaled using the downstream `ConstraintUpdate` concrete type: `Rescale`.

```@docs; canonical=false
Rescale
```
