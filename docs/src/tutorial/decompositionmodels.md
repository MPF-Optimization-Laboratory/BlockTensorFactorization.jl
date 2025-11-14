# Decomposition Models

The main abstract type for all decomposition models is `AbstractDecomposition`.

```@docs; canonical = false
AbstractDecomposition
```

This can be subtyped to create your own decomposition model.

The following common tensor models are available as valid arguments to `model` and built into this package.

### Tucker types

```@docs; canonical = false
Tucker
Tucker1
CPDecomposition
```

Note these are all subtypes of `AbstractTucker`.

```@docs; canonical = false
AbstractTucker
```

### Other types

```@docs; canonical = false
GenericDecomposition
SingletonDecomposition
```

## How Julia treats an AbstractDecomposition

`AbstractDecomposition` is an abstract subtype of `AbstractArray`. `AbstractDecomposition` will keep track of the element type and number of dimensions like other `AbstractArray`. This is the `T` and `N` in the type `Array{T,N}`. To make `AbstractDecomposition` behave like other array types, Julia only needs to know how to access/compute indexes of the array through `getindex`. These indices are computed on the fly when a particular index is requested, or the whole tensor is computed from its factors through `array(X)`. This has the advantage of minimizing the memory used, and allows for the most flexibility since any operation that is supported by `AbstractArray` will work on `AbstractDecomposition` types. The drawback is that repeated requests to entries must recompute the entry each time. In these cases, it is best to "flatten" the array with `array(X)` first before making these repeated calls.

Some basic operations like `+`, `-`, `*`, `\`, and `/` will either compute the operation is some optimized way, or call `array(X)` function to first flatten the decomposition into a regular `Array` type in some optimized way. Operations that don't have an optimized method (because I can only do so much), will instead call Julia's `Array{T,N}(X)` to convert the model into a regular `Array{T,N}` type. This is usually slower and less memory efficient since it calls `getindex` on every index individually, instead of computing the whole array at once.
