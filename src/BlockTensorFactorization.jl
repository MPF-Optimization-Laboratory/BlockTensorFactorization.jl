module BlockTensorFactorization

include("./Core/Core.jl")

using .Core

# Basic functionality
#include("./utils.jl")
export SuperDiagonal, abs_randn, all_recursive, eachfibre, getnotindex, geomean, identityslice, interlace, multifoldl, norm2, proj_one_hot, projsplx, proj_one_hot!, projsplx!, reshape_ndims

#include("./curvaturetools.jl")
export d_dx, d2_dx2, curvature, standard_curvature

#include("./tensorproducts.jl")
export nmp, nmode_product, mtt, slicewise_dot, tuckerproduct, cpproduct, outer_product
export ×₁, ×₂, ×₃, ×₄, ×₅, ×₆, ×₇, ×₈, ×₉
export ⋅₁, ⋅₂, ⋅₃, ⋅₄, ⋅₅, ⋅₆, ⋅₇, ⋅₈, ⋅₉

# Low level types and interface
#include("./decomposition.jl")
export array, contractions, core, eachfactorindex, eachrank1term, factor, factors, frozen, isfrozen, matrix_factor, matrix_factors, nfactors, rankof
export AbstractDecomposition
export GenericDecomposition, SingletonDecomposition
export AbstractTucker, Tucker, Tucker1, CPDecomposition

#include("./objective.jl")
export AbstractObjective
export L2

#include("./factormatching.jl")
export match_cols!, match_rows!, match_slices!
export match_factors!

#include("./constraint.jl")
export AbstractConstraint
export check

export GenericConstraint
export NoConstraint, noconstraint
export ComposedConstraint

export ProjectedNormalization
export l2normalize!, l2normalize_rows!, l2normalize_cols!, l2normalize_1slices!, l2normalize_12slices!
export l1normalize!, l1normalize_rows!, l1normalize_cols!, l1normalize_1slices!, l1normalize_12slices!
export linftynormalize!, linftynormalize_rows!, linftynormalize_cols!, linftynormalize_1slices!, linftynormalize_12slices!
export simplex!, simplex_rows!, simplex_cols!, simplex_1slices!, simplex_12slices!

export ScaledNormalization
export l2scale!, l2scale_rows!, l2scale_cols!, l2scale_1slices!, l2scale_12slices!
export l1scale!, l1scale_rows!, l1scale_cols!, l1scale_1slices!, l1scale_12slices!
export linftyscale!, linftyscale_rows!, linftyscale_cols!, linftyscale_1slices!, linftyscale_12slices!
export l1scale_average12slices!, l2scale_average12slices!, linftyscale_average12slices!

export BUILT_IN_CONSTRAINTS

export Entrywise, IntervalConstraint
export nonnegative!, binary!, binaryproject
export l1norm, l2norm, linftynorm
export isnonnegative_sumtoone

export LinearConstraint

#include("./stats.jl")
export AbstractStat
export DisplayDecomposition, EuclideanLipschitz, EuclideanStepSize, FactorNorms, GradientNorm, GradientNNCone
export IterateNormDiff, IterateRelativeDiff, Iteration, ObjectiveValue, ObjectiveRatio, PrintStats, RelativeError

#include("./blockupdates.jl")
export AbstractStep
export LipschitzStep, ConstantStep, SPGStep

export AbstractUpdate
export GradientDescent, MomentumUpdate

export ConstraintUpdate, GenericConstraintUpdate, IdentityUpdate
export Projection, NNProjection, SafeNNProjection, Rescale

export BlockedUpdate
export updates
export getconstraint, smart_insert!, smart_interlace!, group_by_factor

#export block_gradient_decent, nn_block_gradient_decent, scale_nn_block_gradient_decent, proj_nn_block_gradient_decent

# High level / user-interface
#include("./factorize.jl")
export factorize, default_kwargs

#include("./multiscale.jl")
export multiscale_factorize
export coarsen, interpolate, scale_constraint

#include("./rankdetection.jl")
export rank_detect_factorize
export max_possible_rank

end # module BlockTensorFactorization
