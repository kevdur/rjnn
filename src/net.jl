#==============================================================================#
# RJNN: Neural networks
#==============================================================================#

#==============================================================================#
# Notes:
# 1. I use terminology similar to that found in Bishop's 'Pattern Recognition
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
#  layers l and l+1 of a network are stored in a matrix of size m x n, where m
#  and n are the sizes of layers l and l+1 respectively. Activations are
#  computed via left multiplication (with the weight matrix appearing on the
#  right). The bias terms for each layer are held in row vectors; separating
#  them from the weight matrices in this way makes array handling slightly
#  simpler.
# 4. The activation function used here is f(x) = 1.7159 tanh(2x/3), as
#  recommended by LeCun ('Efficient BackProp', 1998). It is symmetric about the
#  x-axis and, when used with normalised inputs, has an output variance of
#  roughly 1 (because f(±1) ≈ ±1).
# 5. The gradient of the weight parameters is stored as part of the network
#  object; it contains one value per weight parameter (the derivative of the
#  total prediction error with respect to that parameter). Since the total error
#  is the sum of per-observation errors, the derivatives that make up the
#  gradient can be computed by summing simpler, per-observation derivatives.
#  Each of these computations (of the gradient based on a single observation)
#  involves both a feedforward and a backpropagation pass, so it makes sense to
#  store the activities computed during the forward pass for use during
#  backpropagation. Ideally we would like to compute the activities (one per
#  unit) for all of the observations at once, in a vectorised way, but storing
#  all of these values is not feasible. As a compromise, we compute and store
#  activities in batches (of observations); these activities are then used to
#  compute per-observation derivatives, which are added to the gradient vector.
#
# To do:
# 1. Apply PCA to the data in addition to standardisation.
# 2. Try different activation functions (ELU, SELU, etc.; see Keras). Consider
#  RJMCMC moves that change the activation function of a network/layer?
# 3. Only use one matrix for backpropagation errors.
#==============================================================================#

using LinearAlgebra

"An immutable neural network, containing both architecture and parameters."
struct Net{T<:AbstractMatrix{Float64}}
    # Weight-related matrices are type parameterised to allow for the use of
    # sparse (or other) matrix types.
    weights::Vector{T} # weights[l] connect l and l+1.
    biases::Vector{Matrix{Float64}} # per-unit biases, as row vectors.
    gradients::Vector{T} # per-weight error derivatives.
    activities::Vector{Matrix{Float64}} # batch feedforward activities.
    errors::Vector{Matrix{Float64}} # batch backpropagation errors.
end

function Net(weights, biases, batch_size)
    gradients = [similar(W) for W in weights]
    activities = Vector{Matrix{Float64}}(undef, length(weights) + 1)
    errors = Vector{Matrix{Float64}}(undef, length(weights) + 1)
    net = Net(weights, biases, gradients, activities, errors)
    batchsize!(net, batch_size)
    net
end

# The input layer does not count towards the depth (number of layers) of the
# network. It is handled by the width function for convenience sake, via the
# index l = 0.
inputs(net) = size(net.weights[1], 1)
outputs(net) = size(net.weights[end], 2)
depth(net) = length(net.weights)
width(net, l) = l == 0 ? inputs(net) : size(net.weights[l], 2)
batchsize(net) = size(net.activities[1], 1)

"Updates a network to operate with a given (maximum) batch size."
function batchsize!(net, batch_size)
    for l = 0:depth(net)
        w = width(net, l)
        net.activities[l+1] = Matrix{Float64}(undef, batch_size, w)
        net.errors[l+1] = Matrix{Float64}(undef, batch_size, w)
    end
end

function Base.show(io::IO, net::Net)
    print(io, "$(depth(net))-layer Net: $(inputs(net)) x ")
    join(io, (repr(width(net, l)) for l in 1:depth(net)), " x ")
end

function Base.show(io::IO, ::MIME"text/plain", net::Net)
    print(io, net, "\n")
    io = IOContext(io, :compact=>true, :limit=>true)
    for l in 1:depth(net)
        print(io, "$l: ", net.weights[l], " + ", net.biases[l], '\n')
    end
end