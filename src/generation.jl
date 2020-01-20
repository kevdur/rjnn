#==============================================================================#
# RJNN: Network and Data Generation
#==============================================================================#

using Random

"Returns a neural network with random edges and weights."
function randnet(ins::Int, outs::Int, hidden::Vector{Int}, density=0.5::Real)
    n_layers = 2 + length(hidden)
    n_nodes = [ins, hidden..., outs]
    weights = [zeros(m, n) for (m, n) in zip(n_nodes[1:end-1], n_nodes[2:end])]
    biases = zeros(n_layers-1)

    # Initialise a random subset of the weights and biases, and ensure that
    # every node is connected to its neighbouring layers by at least one edge --
    # i.e., that the weight matrices have no empty rows or columns.
    for ws in weights
        _randnfill!(ws, density)
        _randnconnect!(ws)
    end
    _randnfill!(biases, density)

    Net(n_layers, n_nodes, weights, biases)
end

"Initialises a random subset of a matrix's elements using a standard normal
 distribution."
function _randnfill!(A::AbstractArray{Float64}, p::Real)
    js = randsubseq(eachindex(A), p) # each position is chosen with independent
    randn!(@view A[js])              # probability p (Bernoulli sampling).
    nothing
end

"Ensures that every row and column of a matrix contains at least one non-zero
 value."
function _randnconnect!(A::AbstractMatrix{Float64})
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
    nothing
end