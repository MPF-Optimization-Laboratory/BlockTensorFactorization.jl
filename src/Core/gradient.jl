"""
Calculates the gradients, and Lipschitz constants for arbitrary objectives
"""

#---------Manual Lipschitz Constant----------#

# TODO have these be functions that act on decompositions more generally

function make_lipschitz(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function lipschitz0(T::Tucker1; kwargs...)
            A = matrix_factor(T, 1)
            return opnorm(A'A)
        end
        return lipschitz0

    elseif n==1 # the matrix is the zeroth factor
        function lipschitz1(T::Tucker1; kwargs...)
            C = core(T)
            return opnorm(slicewise_dot(C, C))
        end
        return lipschitz1

    else
        error("No $(n)th factor in Tucker1")
    end
end

function make_lipschitz(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipschitz_core(T::AbstractTucker; kwargs...)
            #matrices = matrix_factors(T)
            #gram_matrices = map(A -> A'A, matrices)
            #return prod(opnorm.(gram_matrices))
            return prod(A -> opnorm(A'A), matrix_factors(T))
        end
        return lipschitz_core

    elseif n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(T::AbstractTucker; kwargs...)
            matrices = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), matrices; exclude=n)
            return opnorm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

function make_lipschitz(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(T::AbstractTucker; kwargs...)
            matrices = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), matrices; exclude=n) # TODO optimize this to avoid making the super diagonal core
            return opnorm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in CPDecomposition")
    end
end

function make_block_lipschitz(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function lipschitz0(T::Tucker1; kwargs...)
            A = matrix_factor(T, 1)
            return Diagonal_col_norm(A'A)#Diagonal(A'A)# # Diagonal(norm2.(eachcol(A)))
        end
        return lipschitz0

    elseif n==1 # the matrix is the zeroth factor
        function lipschitz1(T::Tucker1; kwargs...)
            C = core(T)
            return Diagonal_col_norm(slicewise_dot(C, C))#Diagonal(slicewise_dot(C, C))# #Diagonal(norm2.(eachslice(C; dims=1)))
        end
        return lipschitz1

    else
        error("No $(n)th factor in Tucker1")
    end
end

function make_block_lipschitz(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(T::AbstractTucker; kwargs...)
            matrices = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), matrices; exclude=n) # TODO optimize this to avoid making the super diagonal core
            return Diagonal_col_norm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

function make_block_lipschitz(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipschitz_core(T::AbstractTucker; kwargs...)
            return map(A -> Diagonal_col_norm(A'A), matrix_factors(T)) # Return a tuple of diagonal matrices
        end
        return lipschitz_core

    elseif n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(T::AbstractTucker; kwargs...)
            matrices = matrix_factors(T)
            TExcludeAn = tuckerproduct(core(T), matrices; exclude=n)
            return Diagonal_col_norm(slicewise_dot(TExcludeAn, TExcludeAn; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

#--------Manual Gradient--------#

function make_gradient(D::AbstractDecomposition, n::Integer, Y::AbstractArray; objective::AbstractObjective, kwargs...)
    # error("Gradient not implemented for ", typeof(D), " with ", typeof(objective), " objective")
    
    decomposition_type = typeof(D)

    function f(factors) # can only auto-diff arrays, tuples, and tuples of arrays
        decomposition = build_decomposition(decomposition_type, factors)
        return objective(decomposition, Y) # converted tuple of arrays to a decomposition type
    end

    f_tape = GradientTape(f, factors(D))
    compiled_f_tape = compile(f_tape) #TODO could be more efficient by only using one complied tape, 
                                      #rather than a new one for each factor n
    factor_n = eachfactorindex(X)[n]

    function ∇f_n(X; kwargs...)
        G = gradient(compiled_f_tape, factors(X))
        # return G[n] # the nth factor may not be the nth element! e.g. 0th factor for a tucker1
        return G[factor_n]
    end

    return ∇f_n
end

# Using this pattern of inputs so that gradients for a generic decomposition could be calculated
# with auto diff by looking at the gradient of the function objective(D, Y) with respect to the nth factor in D
function make_gradient(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function gradient0(X::Tucker1; kwargs...)
            (B, A) = factors(X)
            AA = A'A
            YA = Y×₁A'
            grad = B×₁AA - YA
            return grad
        end
        return gradient0
    elseif n==1 # the matrix is the first factor
        function gradient1(X::Tucker1; kwargs...)
            (B, A) = factors(X)
            BB = slicewise_dot(B, B)
            YB = slicewise_dot(Y, B)
            grad = A*BB - YB
            return grad
        end
        return gradient1
    else
        error("No $(n)th factor in Tucker1")
    end
end

function make_gradient(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function gradient_core(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            gram_matrices = map(A -> A'A, matrices) # gram matrices AA = A'A,
                                                    # BB = B'B, ...
            grad = tuckerproduct(B, gram_matrices) - tuckerproduct(Y, adjoint.(matrices))
            return grad
        end
        return gradient_core

    elseif n in 1:N # the matrix factors start at m=1
        function gradient_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            Aₙ = factor(X, n)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            grad = Aₙ * slicewise_dot(X̃ₙ, X̃ₙ; dims=n) - slicewise_dot(Y, X̃ₙ; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end

function make_gradient(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix factors start at m=1
        function gradient_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            Aₙ = factor(X, n)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            grad = Aₙ * slicewise_dot(X̃ₙ, X̃ₙ; dims=n) - slicewise_dot(Y, X̃ₙ; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end