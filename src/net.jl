#==============================================================================#
# RJNN: Neural Networks
#==============================================================================#

#==============================================================================#
# Notes:
# 1. We use terminology similar to that found in Bishop's 'Pattern Recognition
#  and Machine Learning' (Chapter 5). In particular, the input to a neuron's
#  activation function is called the 'activation', the resulting output is
#  called the 'activity' (this is MacKay's choice of terminology), and the
#  derivatives of the error function with respect to the activations (used
#  during backpropagation) are simply called 'errors'.
# 2. Data sets are stored as matrices in which the columns and rows correspond
#  to features and observations respectively; a data set containing a single
#  observation will be stored as a row vector. These matrices should always be
#  normalised column-wise (so that each column has a mean of 0 and a variance of
#  1) before being used in training, prediction, etc.
# 3. To complement the structure of the data matrix, the weights connecting
#  layers l and l+1 of a network are stored in a matrix of size m+1 x n, where m
#  and n are the sizes of layers l and l+1 respectively. The final row of each
#  matrix holds bias terms. Activations are computed via left multiplication
#  (with the weight matrix appearing on the right).
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

"An immutable neural network, containing both architecture and parameters."
struct Net
    n_layers::Int # input, hidden, and output.
    n_nodes::Vector{Int} # number of units in each layer.
    weights::Vector{AbstractMatrix{Float64}} # weights[l] connect l and l+1.
end

"Applies a neural network to a given data set."
function predict!(net::Net, data::AbstractMatrix{Float64},
        A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    # A and B are auxiliary matrices that will be overwritten -- the result will
    # be stored in one of them. Both matrices should have the same number of
    # rows as the data matrix, and at least as many columns as the tallest
    # weight has rows. Two auxiliary matrices are required because the in-place
    # matrix multiplication operation cannot store its result in one of its
    # arguments, and we need to use the output of each iteration as input to the
    # next.
    A[:, 1:size(data, 2)] = data
    for W in net.weights
        activities!(A, W, B)
        A, B = B, A
    end
    A
end

"Computes the activities of the nodes in a single layer."
function activities!(A::AbstractMatrix{Float64}, W::AbstractMatrix{Float64},
        B::AbstractMatrix{Float64})
    # A holds the activities of the previous layer, W the weights connecting the
    # layers, and B is an auxiliary matrix in which the result will be stored.
    m, n = size(W)
    A[:,m] .= 1
    Av, Bv = @views A[:,1:m], B[:,1:n]
    mul!(Bv, Av, W, 1, 0) # computes A*W + B.
    Bv .= 1.71590471 .* tanh.(0.66666667 .* Bv)
end