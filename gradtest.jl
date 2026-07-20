using BlockTensorFactorization, Random, ReverseDiff, BenchmarkTools, LinearAlgebra

n,r = 100, 3
D = CPDecomposition((n,n,n),r)
Y = randn(n,n,n)

g_auto = BlockTensorFactorization.Core.make_gradient(D, 1, Y, Lp(2))
g_manual1 = BlockTensorFactorization.Core.make_gradient(D, 1, Y, L2())
g_manual2 = BlockTensorFactorization.Core.make_gradient(D, 2, Y, L2())
g_manual3 = BlockTensorFactorization.Core.make_gradient(D, 3, Y, L2())
g_full1(X) = (g_manual1(X), g_manual2(X), g_manual3(X))[1]

@btime g_auto(D)
@btime g_manual1(D)
@btime g_full1(D)

#=
Results for n,r= 100,3

422.188 ms (11 allocations: 7.54 KiB)
  1.906 ms (4223 allocations: 692.24 KiB)
  7.541 ms (12672 allocations: 2.03 MiB)

Auto grad does use less memory!
but it's much slower
=#

# ---------------

struct L2_alt <: AbstractObjective 
end

quick_norm_diff(X, Y) = begin
    X = array(X)
    total = 0
    for i in eachindex(Y)
        total += (X[i]-Y[i])^2
    end
    return total
end

function (objective::L2_alt)(X, Y)
    # return norm(X - Y, 2)^2 # TODO should this be 1/p ? I think not since p could be Inf
    # return 1 # dead fast, so how the gradient is calculated maters
    return norm2(X - Y)
    # return quick_norm_diff(X, Y)
end


g_auto_alt = BlockTensorFactorization.Core.make_gradient(D, 1, Y, L2_alt())
@btime g_auto_alt(D)

# ------------- 

mynormp1(X, p) = sum(x -> x^p, X)
mynormp2(x) = mapreduce(abs2, +, x)
mynormp3(X) = mapreduce(x -> x^2, +, X)
mynormp4(X, p) = mapreduce(x -> abs(x)^p, +, X)
mynormp5(X, p) = mapreduce(x -> x^p, +, X)
mynormp6(X, p) = mapreduce(x -> Base.literal_pow(^, x, Val(p)), +, X)
make_mynormp(p) = X -> mapreduce(x -> x^p, +, X)
mynormp7(X) = make_mynormp(2)(X)
mynormp8 = make_mynormp(2)

@btime mynormp1(X, 2) setup=(X=randn(100))
@btime mynormp2(X) setup=(X=randn(100))
@btime mynormp3(X) setup=(X=randn(100))
@btime mynormp4(X, 2) setup=(X=randn(100))
@btime mynormp5(X, 2) setup=(X=randn(100))
@btime mynormp6(X, 2) setup=(X=randn(100))
@btime mynormp7(X) setup=(X=randn(100))
@btime mynormp8(X) setup=(X=randn(100))
@btime norm(X)^2 setup=(X=randn(100))
@btime norm(X, 2)^2 setup=(X=randn(100))


# ------------- 

mynormp1(X) = sum(abs, X)
mynormp2(x) = mapreduce(abs, +, x)
mynormp3(X) = norm(X, 1)

@btime mynormp1(X) setup=(X=randn(100))
@btime mynormp2(X) setup=(X=randn(100))
@btime mynormp3(X) setup=(X=randn(100))

mynormp1(X) = sum(abs2, X)
mynormp2(x) = mapreduce(abs2, +, x)
mynormp3(X) = norm(X, 2)

@btime mynormp1(X) setup=(X=randn(100))
@btime mynormp2(X) setup=(X=randn(100))
@btime mynormp3(X) setup=(X=randn(100))

# ----------

using Zygote

n,r = 10, 3
D = CPDecomposition((n,n,n),r)
Y = randn(n,n,n)

decomposition_type = typeof(D)
objective = L2()

function my_f(factors...)
    decomposition = build_decomposition(decomposition_type, factors)
    return objective(decomposition, Y)
end

f_tape = ReverseDiff.GradientTape(my_f, factors(D))
compiled_f_tape = ReverseDiff.compile(f_tape)
function ∇f_rev(X; kwargs...)
    G = ReverseDiff.gradient!(compiled_f_tape, factors(X))
    return G[1]
end

function ∇f_zygote(X; kwargs...)
    G = Zygote.gradient(my_f, factors(X)...)
    return G[1]
end

∇f_rev(D)
∇f_zygote(D)

using Zygote: @adjoint

@adjoint CPDecomposition{Float64, 3}(factors, frozen) = CPDecomposition{Float64, 3}(factors, frozen), x -> begin
    zero.(factors)
end

#---------------

using BenchmarkTools
using Random

size = (1000,)

function simplex_rand(size)
    A = abs.(randn(size))
    A ./= sum(A)
    return A
end
simplex_rand(size...) = simplex_rand(size)

KL1(X, Y) = sum(@. X * log(X / Y))
KL2(X, Y) = begin
    total = 0
    for (x, y) in zip(X, Y)
        total += x * log(x / y)
    end
    total
end
KL3(X, Y) = begin
    total = 0
    for i in eachindex(X)
        total += X[i] * log(X[i] / Y[i])
    end
    total
end

@btime KL1(X,Y) setup=(X = simplex_rand(size);Y= simplex_rand(size))
@btime KL2(X,Y) setup=(X = simplex_rand(size);Y= simplex_rand(size))
@btime KL3(X,Y) setup=(X = simplex_rand(size);Y= simplex_rand(size))

X = simplex_rand(size)
Y = simplex_rand(size)

grad_tape1 = ReverseDiff.GradientTape(x -> KL1(x, Y), X) |> ReverseDiff.compile
grad_tape2 = ReverseDiff.GradientTape(x -> KL2(x, Y), X) |> ReverseDiff.compile
grad_tape3 = ReverseDiff.GradientTape(x -> KL3(x, Y), X) |> ReverseDiff.compile

G = zero(X)

@btime ReverseDiff.gradient!(grad_tape1, X) setup=(X = simplex_rand(size));
@btime ReverseDiff.gradient!(grad_tape2, X) setup=(X = simplex_rand(size));
@btime ReverseDiff.gradient!(grad_tape3, X) setup=(X = simplex_rand(size));

################

using BenchmarkTools
using Random

abs1(x) = abs(x)
abs2(x) = sign(x) * x
abs3(x) = x ≥ 0 ? x : -x

@btime abs1.(x) setup=(x = randn(1000))
@btime abs2.(x) setup=(x = randn(1000))
@btime abs3.(x) setup=(x = randn(1000));
