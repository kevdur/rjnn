#==============================================================================#
# RJNN: Random network and data generation
#==============================================================================#

using Random

"Returns a neural network with random edges and weights."
function randnet(ins::Integer, outs::Integer, hidden::Vector{<:Integer};
        batch_size=1024, density=0.5)
    sizes = [ins, hidden..., outs]
    dims = zip(sizes[1:end-1], sizes[2:end])
    weights = [zeros(m, n) for (m, n) in dims]
    biases = [zeros(1, n) for n in sizes[2:end]]

    # Initialise a random subset of the weights and biases, and ensure that
    # every unit is connected to its neighbouring layers by at least one edge --
    # i.e., that the weight matrices have no empty rows or columns.
    for (W, b) in zip(weights, biases)
        randnfill!(W, density)
        randnfill!(b, density)
        randnconnect!(W)
    end

    Net(weights, biases, batch_size)
end

"Initialises a random subset of an array's elements using a standard normal
 distribution."
function randnfill!(A::AbstractArray{<:AbstractFloat}, p)
    js = randsubseq(eachindex(A), p) # each position is chosen with independent
    randn!(@view A[js])              # probability p (Bernoulli sampling).
end

"Ensures that every row and column of a matrix contains at least one non-zero
 value."
function randnconnect!(A::AbstractMatrix{<:AbstractFloat})
    for i in axes(A, 1)
        if all(iszero, @view A[i,:])         # we could call iszero on the array
            A[i, rand(axes(A, 2))] = randn() # directly, but it doesn't
        end                                  # short-circuit.
    end
    for j in axes(A, 2)
        if all(iszero, @view A[:,j])
            A[rand(axes(A, 1)), j] = randn()
        end
    end
end