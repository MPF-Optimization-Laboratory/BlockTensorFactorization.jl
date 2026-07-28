using BlockTensorFactorization, Plots, Random, ForwardDiff, ReverseDiff, BenchmarkTools, LinearAlgebra

### ----------------- MAY 12th - MAY 19th -----------------------
### ------------------------ Start ------------------------------

### its on colab, but creating problems here

### ----------------- MAY 12th - MAY 19th -----------------------
### ------------------------- End -------------------------------

### ----------------- MAY 19th - MAY 26th -----------------------
### ------------------------ Start ------------------------------

"""
QUESTIONS:

1. Is using auto diff slower or more memory intensive? You can check 
with time or benchmark (from the BenchmarkTools package)

2. Is there a special way we need to use auto diff so that the 
performance is similar? (in-place updates, treat the variables as a
single vector v=[x, y], freeze or select variables, compute the full
gradient with respect to x and y separately, or jointly, compile the
gradient tape etc.)
"""


# manually getting the gradient
function grad_f(x, y, q=1e4)
    a = x^2 + (y-2)^2
    b = (x-2)^2 + y^2

    dx = (q^a * (2*x) + q^b * (2*(x-2))) / (q^a + q^b)
    dy = (q^a * (2*(y-2)) + q^b * (2*y)) / (q^a + q^b)

    return dx, dy
end


function bcd_manual(v, q, h)
    for i in 1:500
        # new x
        dx, i = grad_f(v[1], v[2], q)
        v[1] = v[1] - h * dx
        # new y
        i, dy = grad_f(v[1], v[2], q)
        v[2] = v[2] - h * dy
    end
    return v
end

# test to see if close to (1,1)
bcd_manual([5.0, -2.0], 1e4, 0.05)
println("Near (1,1)? ", v)
println("---------------------")
# time
@btime bcd_manual([5.0, -2.0], 1e4, 0.05)
println("---------------------")
"""
ANSWERS:

1. We can see that `bcd_manual` is faster in terms of run time and 
it's less memory intensive.

2. The run times are quite similar, however the memory intensiveness 
is different. We could try to reduce the number of allocations. So 
maybe instead of creating a vector like `[v[1], y]`, inside the calls 
for derivative
(EX: `ForwardDiff.derivative(x -> smoothmax([x, v[2]], q), v[1])`), 
we could try to use something like `[v, y]` where 
instead of a number (`v[1]`), we pass through a vector (`v`).
"""
### ----------------- MAY 19th - MAY 26th -----------------------
### ------------------------- End -------------------------------

### ----------------- JUNE 2nd - JUNE 9th -----------------------
### ------------------------ Start ------------------------------


function smoothmax_v(v, q=1e4)
    x, y = v[1], v[2]
    a = x^2 + (y-2)^2
    b = (x-2)^2 + y^2
    return log(q^a + q^b) / log(q)
end

function bcd_optimized!(v, q, h, grad_new)
    ForwardDiff.gradient!(grad_new, x -> smoothmax_v(x, q), v)

    v[1] -= h * grad_new[1]
    v[2] -= h * grad_new[2]

    return v
end

# test
v = [5.0, -2.0]
grad_new = [0.0,0.0]
h = 0.005
q = 1e4

for i in 1:500
    bcd_optimized!(v, q, h, grad_new)
end

println("Final vector: ", v)
println("Near (1,1)? ", isapprox([1.0, 1.0], v, atol = 0.03))

# memory
println("------ Benchmark Results ------")
@btime bcd_optimized!(v, 1e4, 0.005, grad_new)

"""
I reduced the number of allocations and bytes used 
(from *6 allocations: 272 bytes* to *5 allocations: 192 bytes*) 
by using something like `[v, y]` where instead of a number, I passed 
through a vector. BUT I increased the time by around *200 ns* :( . 
Also from last time, pre computing the gradient helped a lot with the 
memory allocation.
"""
### ----------------- JUNE 2nd - JUNE 9th -----------------------
### ------------------------- End -------------------------------

### ----------------- JUNE 9nd - JUNE 23rd -----------------------
### ------------------------ Start ------------------------------

# real a and real b
a_real = randn(3) ; b_real = randn(3)
Y = a_real * b_real'

# initialize
a = randn(3) ; b = randn(3)

#f
function f(a, b)
    res = a*b' - Y
    return 0.5 * norm(res)^2
end

# manually compute gradient
function grad_manual(a, b)
    res = a*b' - Y
    ga = res * b
    gb = a' * res
    return ga, gb
end
### make separate grad_man_a and grad_man_b

# reversediff (rd)

flat(x) = f(x[1:3], x[4:6])

# rd gradient ###(need 1 at a time)
v = [a; b]
grad_rd = ReverseDiff.gradient(flat, v)

# manual gradient
ga_man, gb_man = grad_manual(a, b)

println("Manual grad (a): ", ga_man)
println("ReverseDiff grad (a): ", grad_rd[1:3])
println(a_real)
println("________________________________________________")
println("Manual grad (b): ", gb_man)
println("ReverseDiff grad (b): ", grad_rd[4:6])
println(b_real)
println("________________________________________________")
# memory
println("memory n time")
@btime grad_manual(a, b)

### ----------------- JUNE 9nd - JUNE 23rd -----------------------
### ------------------------- End --------------------------------

### ----------------- JUNE 23rd - JUNE 30th -----------------------
### ------------------------ Start -------------------------------


# real a and real b
a_real = randn(3) ; b_real = randn(3)
Y = a_real * b_real'

# initialize
a = randn(3) ; b = randn(3)

#f
function f(a, b)
    res = a*b' - Y
    return 0.5 * norm(res)^2
end


# bcd_auto
function bcd_auto(f, a, b; ε=1e-8)
    grad_a = zero(a) ; grad_b = zero(b) ; grad_norm = Float64[]

    while true
        grad_a = ReverseDiff.gradient(x -> f(x, b), a)
        a = a - 0.01 * grad_a

        grad_b = ReverseDiff.gradient(x -> f(a, x), b)
        b = b - 0.01 * grad_b

        if f(a, b) < ε
            break
        end
        #if possible could you remind me why we use append here
        append!(grad_norm, norm(grad_a))
    end
    return a, b, grad_norm
end

# bcd_auto!
function bcd_auto!(f, a, b; ε=1e-8)
    grad_a = zero(a) ; grad_b = zero(b) ; grad_norm = Float64[]

    while true
        ReverseDiff.gradient!(grad_a, x -> f(x, b), a)
        grad_a .*= 0.01
        a .-= grad_a

        ReverseDiff.gradient!(grad_b, x -> f(a, x), b)
        grad_b .*= 0.01
        b .-= grad_b

        if f(a, b) < ε
            break
        end
    end
    return a, b
end

# manually getting gradients
function grad_f_a(a, b, Y)
    return (a * b' - Y) * b
end

function grad_f_b(a, b, Y)
    return (b * a' - Y') * a
end

# bcd_manual
function bcd_manual!(f, a, b, Y; ε=1e-8)
    grad_a = zero(a)
    grad_b = zero(b)

    while true
        grad_a .= grad_f_a(a, b, Y)
        grad_b .= grad_f_b(a, b, Y)

        a .-= 0.01 .* grad_a
        b .-= 0.01 .* grad_b

        if f(a, b) < ε
            break
        end
    end
    return a, b
end

println("time n space")
println("_________________________________________________")

println("bcd_auto")
@btime bcd_auto(f, a, b; ε=1e-8)
println("_________________________________________________")

println("bcd_auto!")
@btime bcd_auto!(f, a, b; ε=1e-8)
println("_________________________________________________")

println("bcd_manual!")
@btime bcd_manual!(f, a, b, Y; ε=1e-8)

"""
**Ranked by efficiency:**

bcd_manual! > bcd_auto! > bcd_auto

in both time and allocations, which is what we expected
"""
### ----------------- JUNE 23rd - JUNE 30th -----------------------
### -------------------------- End --------------------------------

### ----------------- JUNE 30th - JULY 7th -----------------------
### ------------------------ Start -------------------------------

#  Larger Matrix Example

# Same as the previous example, but now we are factorizing $Y$ into two 
# smaller matrices $AB^\top$.

# I also added another version of BCD where we "precompute the gradient 
# tape". This lets Julia record how the output value of $f$ depends on 
# the input variables.

# We have to be a bit creative because the gradient $\nabla_a f$ of $f$ 
# with respect to $a$ depends on both $a$ AND $b$. So we compute the 
# full gradient $(\nabla_a f(a, b), \nabla_b f(a, b))$, and only use 
# the first half to update $a$ (and the second half to update $b$). 
# We have to compute this twice each iteration, because $a$ has changed 
# by the time we need to use the gradient $\nabla_b f(a,b)$ to update $b$.

# Lastly, rather than a fixed stepsize, we use a changing stepsize that 
# can guarantee the function value will decrease. In general, for twice 
#     differentiable functions, we can use the inverse of the operator 
#     norm (largest eigenvalue) of the Hessian $\nabla_a ^2 f(a,b)$.

# For $f(A,B) = 0.5 \lVert A B^\top - Y \rVert^2$, the gradient with 
# respect to $A$ is 
# $\nabla_A f(A, B) = (A B^\top - Y) B = A B^\top B - Y B$. 
# Taking another derivative with respect to $A$ gives us 
# $\nabla_A^2 f(A, B) = B^\top B$. 
# So we should use a stepsize of $1/ \lVert B^\top B \rVert_\text{op}$.

# Aside 1: really 
# $\nabla_A^2 f(A, B)$ should be the order-4 tensor $B^\top B \otimes I$, 
# but it turns out most of the entries in this tensor are zero. 
# The nonzero entries are equal to $B^\top B$.

# Aside 2: 
# $\lVert B^\top B \rVert_\text{op} = \lVert BB^\top \rVert_\text{op}$ 
# and the second matrix is faster to compute.
 
# real a and real b
m,n,r = 50, 40, 5
a_real = randn(m,r) ; b_real = randn(n,r)
Y = a_real * b_real';


#f
function f(a, b, Y)
    res = a*b' .- Y
    return 0.5 * norm(res)^2
end

# bcd_auto!
function bcd_auto!(f, a, b, Y; ε=1e-8)
    grad_a = zero(a)
    grad_b = zero(b)

    while true
        ReverseDiff.gradient!(grad_a, x -> f(x, b, Y), a)
        a .-= grad_a ./ opnorm(b*b')

        ReverseDiff.gradient!(grad_b, x -> f(a, x, Y), b)
        b .-= grad_b ./ opnorm(a'*a)

        if f(a, b,Y) < ε
            break
        end
    end
    return a, b
end

function bcd_compiled!(f, a, b, Y; ε=1e-8)
    grad_ab = (zero(a), zero(b))
    f_tape = ReverseDiff.GradientTape((x_a, x_b) -> f(x_a, x_b, Y), (a, b))
    compiled_f_tape = ReverseDiff.compile(f_tape)

    while true
        ReverseDiff.gradient!(grad_ab, compiled_f_tape, (a,b))
        a .-= grad_ab[1] ./ opnorm(b*b')

        ReverseDiff.gradient!(grad_ab, compiled_f_tape, (a,b))
        b .-= grad_ab[2] ./ opnorm(a'*a)

        if f(a, b,Y) < ε
            break
        end
    end
    return a, b
end

# manually getting gradients
function grad_f_a(a, b, Y)
    return (a * b' - Y) * b
end

function grad_f_b(a, b, Y)
    return (b * a' - Y') * a
end

# bcd_manual
function bcd_manual!(f, a, b, Y; ε=1e-8)
    grad_a = zero(a)
    grad_b = zero(b)

    while true
        grad_a .= grad_f_a(a, b, Y)
        a .-= grad_a ./ opnorm(b*b')

        grad_b .= grad_f_b(a, b, Y)
        b .-= grad_b ./ opnorm(a'*a)

        if f(a, b,Y) < ε
            break
        end
    end
    return a, b
end

println("time n space")
println("_________________________________________________")

println("bcd_compiled!")
@btime bcd_compiled!(f, a, b, Y; ε=1e-8) setup=(a=randn(m,r); b=randn(n,r))
println("_________________________________________________")

println("bcd_auto!")
@btime bcd_auto!(f, a, b, Y; ε=1e-8) setup=(a=randn(m,r); b=randn(n,r))
println("_________________________________________________")

println("bcd_manual!")
@btime bcd_manual!(f, a, b, Y; ε=1e-8) setup=(a=randn(m,r); b=randn(n,r))

# Look at the function value and gradient convergence
function bcd_manual_with_stats!(f, a, b, Y; ε=1e-8)
    grad_a = zero(a)
    grad_b = zero(b)
    norm_grad_As = Float64[]
    norm_grad_Bs = Float64[]
    function_values = Float64[]

    while true
        grad_a .= grad_f_a(a, b, Y)
        a .-= grad_a ./ opnorm(b*b')

        grad_b .= grad_f_b(a, b, Y)
        b .-= grad_b ./ opnorm(a'*a)

        append!(norm_grad_As, norm(grad_a))
        append!(norm_grad_Bs, norm(grad_b))
        append!(function_values, f(a, b, Y))

        if function_values[end] < ε
            break
        end
    end
    return a, b, (norm_grad_As, norm_grad_Bs, function_values)
end

a, b, stats = bcd_manual_with_stats!(f, randn(m,r), randn(n,r), Y; ε=1e-8)
(norm_grad_As, norm_grad_Bs, function_values) = stats

function my_plot(y; kwargs...)
    p = plot()
    plot!(y;
        xlabel="iteration",
        ylabel="value",
        yaxis=:log10,
        kwargs...
    )
    return p
end

my_plot(norm_grad_As; label="norm grad_a")
my_plot(norm_grad_Bs; label="norm grad_b")
my_plot(function_values; label="function value")

### ----------------- JUNE 30th - JULY 7th -----------------------
### -------------------------- End --------------------------------

### ----------------- JULY 7th - JULY 14th -----------------------
### ------------------------ Start -------------------------------

using Random
using LinearAlgebra
using ReverseDiff
using BenchmarkTools

m, n, r = 5, 5, 3
a_true = randn(m, r)
b_true = randn(n, r)

# Y = a*b' + maybe some noise
Y = a_true * b_true' #+ 0.01 * randn(m, n)

"Function Try #1"

#function kl(a, b, Y)
 # p = a*b'
  #q = Y
  #return sum(p .* log(p ./ q))
#end


"Function Try #2"

function kl(a, b, Y)
    p = a * b'
#to prevent log zero/negative
    p_safe = max.(p, 1e-12)
    y_safe = max.(Y, 1e-12)
    return sum(p_safe .* log.(p_safe ./ y_safe))
end

### not using most recent a and b
### bcd_auto_tape_compile
function bcd_auto_tape_compile!(f, a, b, Y)

    grad_a = zero(a)
    grad_b = zero(b)

    tape_a = ReverseDiff.GradientTape(x -> f(x, b, Y), a)
    compiled_tape_a = ReverseDiff.compile(tape_a)

    tape_b = ReverseDiff.GradientTape(x -> f(a, x, Y), b)
    compiled_tape_b = ReverseDiff.compile(tape_b)

    for i in 1:1000
        ReverseDiff.gradient!(grad_a, compiled_tape_a, a)

        # update but keep a and b > 0
        a .-= 0.1 .* (grad_a ./ (norm(b*b') + 1e-12))
        a .= max.(a, 1e-12)


        ReverseDiff.gradient!(grad_b, compiled_tape_b, b)

        b .-= 0.1 .* (grad_b ./ (norm(a'*a) + 1e-12))
        b .= max.(b, 1e-12)

        if f(a, b, Y) < 1e-8
            break
        end
    end
    return a, b
end

# test
a, b = randn(m, r), randn(n, r)

println("Test w kl and bcd_auto_tape_compile!:")
@btime bcd_auto_tape_compile!(kl, a, b, Y)
@show isapprox(a*b', Y, rtol = 0.01)




# test to see if we are using new a / new b when calling gradient 
# tape/gradient

b_true = randn(3,2)
a_true = randn(3,2)
Y = a_true * b_true'

b = randn(3,2)
a = randn(3,2)

function kl(a, b, Y)
    p = a * b'
#to prevent log zero/negative
    p_safe = max.(p, 1e-12)
    y_safe = max.(Y, 1e-12)
    return sum(p_safe .* log.(p_safe ./ y_safe))
end

g_vec = zeros(3,2)
tape_b = ReverseDiff.GradientTape(x -> kl(a, x, Y), b)
compiled_tape_b = ReverseDiff.compile(tape_b)


println(ReverseDiff.gradient!(g_vec, compiled_tape_b, b))

a = randn(3,2)
ReverseDiff.gradient!(g_vec, compiled_tape_b, b)

### ------------------ JULY 7th - JULY 14th -----------------------
### -------------------------- End --------------------------------

### ----------------- JULY 14th - JULY 21st -----------------------
### ------------------------ Start -------------------------------


function kl(a, b, Y)
    p = a * b'
    # to prevent log zero/negative
    p_safe = max.(p, 1e-12)
    y_safe = max.(Y, 1e-12)
    return sum(p_safe .* log.(p_safe ./ y_safe))
end



### try for backtracking line search
function kl_bcd_compiled_backtracking!(f, a, b, Y)
    grad_ab = (zero(a), zero(b))
    f_tape = ReverseDiff.GradientTape((x_a, x_b) -> f(x_a, x_b, Y), (a, b))
    compiled_f_tape = ReverseDiff.compile(f_tape)

    iter = 0
    while iter < 1000
        iter += 1
        val_current = f(a, b, Y)
        if val_current < 1e-8
            break
        end

        #a
        ReverseDiff.gradient!(grad_ab, compiled_f_tape, (a, b))
        g_a = grad_ab[1]

        alpha_a = 1.0
        c = 1e-4
        g_a_norm_sq = sum(abs2, g_a) #sum of all squared elements in matrix

        a_old = copy(a)
        while alpha_a > 1e-16
            a .= a_old .- alpha_a .* g_a
            if f(a, b, Y) <= val_current - c * alpha_a * g_a_norm_sq
                break
            end
            alpha_a *= 0.5
        end
        a .= max.(a, 0.0) ### tried to eliminate the negativity

        val_current = f(a, b, Y)

        #b
        ReverseDiff.gradient!(grad_ab, compiled_f_tape, (a, b))
        g_b = grad_ab[2]

        alpha_b = 1.0
        g_b_norm_sq = sum(abs2, g_b)

        b_old = copy(b)
        while alpha_b > 1e-16
            b .= b_old .- alpha_b .* g_b
            if f(a, b, Y) <= val_current - c * alpha_b * g_b_norm_sq
                break
            end
            alpha_b *= 0.5
        end
        b .= max.(b, 0.0) ### tried to eliminate the negativity

        if f(a, b, Y) < 1e-8
            break
        end
    end

    return a, b
end


### test
m, n, r = 5, 5, 3
Random.seed!(1)
a_true = abs.(randn(m, r)) ### abs is new
b_true = abs.(randn(n, r)) ### abs is new
Y = a_true * b_true'
Y ./= sum(Y) ### normalizing Y is new

#initialize
a = abs.(randn(m, r))
b = abs.(randn(n, r))

a_opt, b_opt = kl_bcd_compiled_backtracking!(kl, a, b, Y)
println("Should be close to 0 if it worked:   ", kl(a_opt, b_opt, Y))

display(a_true)
display(a_opt)

display(b_true)
display(b_opt)

display(a_opt * b_opt')
display(Y)

### try for secant method


function f_frobenius(a, b, Y)
    res = a * b' .- Y
    #res_safe = max.(res, 1e-12)
    return 0.5 * norm(res)^2
end

function secant_frobenius_bcd!(f, a, b, Y)
    # initialize a
    a_minus_1 = copy(a)
    a_0 = a_minus_1 .+ 0.1 .* randn(size(a))
    a_prev = copy(a_minus_1)
    a_curr = copy(a_0)
    grad_prev_a = zero(a)
    grad_curr_a = zero(a)

    f_tape_a = ReverseDiff.GradientTape(x -> f(x, b, Y), a_prev)
    compiled_tape_a = ReverseDiff.compile(f_tape_a)
    ReverseDiff.gradient!(grad_prev_a, compiled_tape_a, a_prev)

    # initialize b
    b_minus_1 = copy(b)
    b_0 = b_minus_1 .+ 0.1 .* randn(size(b))
    b_prev = copy(b_minus_1)
    b_curr = copy(b_0)
    grad_prev_b = zero(b)
    grad_curr_b = zero(b)

    f_tape_b = ReverseDiff.GradientTape(x -> f(a_curr, x, Y), b_prev)
    compiled_tape_b = ReverseDiff.compile(f_tape_b)
    ReverseDiff.gradient!(grad_prev_b, compiled_tape_b, b_prev)

    iter = 0
    while iter < 10000
        iter += 1

        # check convergence
        if f(a_curr, b_curr, Y) < 1e-8
            a .= a_curr
            b .= b_curr
            break
        end


        # update 'a' (b fixed)
        f_tape_curr_a = ReverseDiff.GradientTape(x -> f(x, b_curr, Y), a_curr)
        compiled_curr_a = ReverseDiff.compile(f_tape_curr_a)
        ReverseDiff.gradient!(grad_curr_a, compiled_curr_a, a_curr)

        denom_a = grad_curr_a .- grad_prev_a
        denom_a .= denom_a .+ 1e-8 .* sign.(denom_a) .+ (abs.(denom_a) .< 1e-8) .* 1e-8
        alpha_k_a = (a_curr .- a_prev) ./ denom_a
        a_next = a_curr .- alpha_k_a .* grad_curr_a

        a_prev .= a_curr
        a_curr .= a_next
        grad_prev_a .= grad_curr_a
        a .= a_curr

        # update 'b' (a fixed)
        f_tape_curr_b = ReverseDiff.GradientTape(x -> f(a_curr, x, Y), b_curr)
        compiled_curr_b = ReverseDiff.compile(f_tape_curr_b)
        ReverseDiff.gradient!(grad_curr_b, compiled_curr_b, b_curr)

        denom_b = grad_curr_b .- grad_prev_b
        denom_b .= denom_b .+ 1e-8 .* sign.(denom_b) .+ (abs.(denom_b) .< 1e-8) .* 1e-8
        alpha_k_b = (b_curr .- b_prev) ./ denom_b
        b_next = b_curr .- alpha_k_b .* grad_curr_b

        b_prev .= b_curr
        b_curr .= b_next
        grad_prev_b .= grad_curr_b
        b .= b_curr
    end

    return a, b
end

# test
m, n, r = 5, 5, 3
Random.seed!(1)
a_true = randn(m, r)
b_true = randn(n, r)
Y = a_true * b_true'

a_minus_1 = randn(m, r)
a_init_val = a_minus_1 .+ 0.1 .* randn(m, r)

b_minus_1 = randn(n, r)
b_init_val = b_minus_1 .+ 0.1 .* randn(n, r)

println("Initial value: ", f_frobenius(a_init_val, b_init_val, Y))

a_opt, b_opt = secant_frobenius_bcd!(f_frobenius, a_init_val, b_init_val, Y)
println("Final value:   ", f_frobenius(a_opt, b_opt, Y))

display(a_true)
display(a_opt)

### ----------------- JULY 14th - JULY 21st -----------------------
### -------------------------- End --------------------------------
