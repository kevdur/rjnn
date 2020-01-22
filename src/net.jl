#==============================================================================#
# RJNN: Neural Networks
#==============================================================================#

#==============================================================================#
# Notes:
# 1. We use notation similar to that found in MacKay's book 'Information Theory,
#  Inference, and Learning Algorithms' (starting in Chapter 39). In particular,
#  the input to a neuron's activation function is called the 'activation', and
#  the resulting output is called the 'activity'.
# 2. Data sets are stored as matrices in which the columns and rows correspond
#  to features and observations respectively; a data set containing a single
#  observation will be stored as a row vector. These matrices should always be
#  normalised column-wise (so that each column has a mean of 0 and a variance of
#  1) before being used in training, prediction, etc.
# 3. To complement the structure of the data matrix, the weights connecting
#  layers l and l+1 of a network are stored in a matrix of size m x n, where m
#  and n are the sizes of layers l and l+1 respectively. Activations are
#  computed via left multiplication (with the weight matrix appearing on the
#  right), and biases are stored as row vectors.
# 4. The activation function used here is f(x) = 1.7159 tanh(2x/3), as
#  recommended by LeCun ('Efficient BackProp', 1998). It is symmetric about the
#  x-axis and, when used with normalised inputs, has an output variance of
#  roughly 1 (because f(±1) ≈ ±1).
#
# To do:
# 1. Apply PCA to the data in addition to standardisation.
# 2. Try different activation functions (ELU, SELU, etc.; see Keras). Consider
#  RJMCMC moves that change the activation function of a network/layer?
#==============================================================================#

using LinearAlgebra

"Immutable neural network, containing both architecture and parameters."
struct Net
    n_layers::Int # input, hidden, and output.
    n_nodes::Vector{Int} # number of units in each layer.
    weights::Vector{AbstractMatrix{Float64}} # weights[l] connect l and l+1.
    biases::Vector{AbstractMatrix{Float64}} # biases[l] are for layer l+1.
end

"Applies a neural network to a given data set."
function predict!(net::Net, data::AbstractMatrix{Float64},
        A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    # A and B are auxiliary matrices that will be overwritten; the result will
    # be stored in A. Both matrices should have the same number of rows as the
    # data matrix, and at least as many columns as the widest weight matrix. Two
    # auxiliary matrices are required because the in-place multiplication
    # operation cannot store its result in one of its arguments, and we need to
    # use the output of each iteration as input to the next.
    A[:, 1:size(data, 2)] = data
    for (W, b) in zip(net.weights, net.biases)
        m, n = size(W)
        Av, Bv = @views A[:,1:m], B[:,1:n]
        Bv .= b # set every row to b.
        mul!(Bv, Av, W, 1, 1) # computes A*W + B.
        Bv .= 1.7159 .* tanh.(0.6667 .* Bv)
        A, B = B, A
    end
    A
end