"""
Benchmarks
"""

using Random
using BenchmarkTools

using BlockTensorFactorization

Random.seed!(3141592653589)

suite = BenchmarkGroup()

suite["factorize"] = BenchmarkGroup(["basic_call"])

C = abs_randn(5, 11, 12)
A = abs_randn(10, 5)
Y = Tucker1((C, A))
Y = array(Y)

my_factorize(Y) = factorize(Y;
    rank=5,
    tolerance=(2, 0.05),
    converged=(GradientNNCone, RelativeError),
    constrain_init=true,
    constraints=nonnegative!,
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError]
)

suite["factorize"][1] = @benchmarkable my_factorize(Y)

tune!(suite)
results = run(suite, verbose = true)

# display(median(results))

BenchmarkTools.save("output.json", median(results))
