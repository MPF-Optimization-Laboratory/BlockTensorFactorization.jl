# Iteration Stats

```@docs; canonical=false
AbstractStat
```

The following stats are supported inputs to the `stats` keyword in factorize.

```@docs; canonical=false
Iteration
GradientNorm
GradientNNCone
ObjectiveValue
ObjectiveRatio
RelativeError
IterateNormDiff
IterateRelativeDiff
EuclidianStepSize
EuclidianLipschitz
FactorNorms
```

The following are subtype of `AbstractStat` but are for auxiliary features.

```@docs; canonical=false
PrintStats
DisplayDecomposition
```
