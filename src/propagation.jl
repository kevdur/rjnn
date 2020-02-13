#==============================================================================#
# RJNN: Forward- and back-propagation
#==============================================================================#

#==============================================================================#
# Notes:
# 1. At the time of writing, @view calls still allocate a small amount of memory
#  on the heap. These calls are necessary when dealing with the final, partial
#  batch of data, which is handled separately from the other batches in the
#  functions below. If views were 'free', we could simply apply them to all of
#  the batches and make the code a bit neater.
#==============================================================================#

h(x) = 1.71590471*tanh.(0.66666667x)
dh(h) = 1.14393647*(1 - 0.33963596h^2) # takes h(x) as an argument.

"Applies a neural network to a given set of observations."
function predict!(net, data::AbstractMatrix{Float64},
        A::AbstractMatrix{Float64})
    # A is an auxiliary matrix that will be used to store the results; it should
    # have as many rows as the data matrix and as many columns as the network's
    # output layer.
    k, n = batchsize(net), size(data, 1)
    for i = 1:k:n
        j = min(i+k-1, n) # the final batch might not be full.
        feedforward!(net, @view data[i:j,:])
        if j == i+k-1
            A[i:j,:] = net.activities[end]
        else
            A[i:n,:] = @view net.activities[end][1:n-i+1,:]
        end
    end
end

"Computes and stores activities for all of a network's units."
function feedforward!(net, data_batch::AbstractMatrix{Float64})
    B, k = data_batch, size(data_batch, 1)
    partial = k < batchsize(net)
    for (W, b, A) in zip(net.weights, net.biases, net.activities)
        if partial
            A = @view A[1:k,:]
        end
        activities!(A, B, W, b)
        B = A
    end
end

"Computes activities for a single layer of units."
function activities!(A, B, W, b)
    # B contains the activities of the previous layer (or inputs), and W and b
    # the weights and biases of the current layer. The resulting activities
    # h(B*W + b) will be stored in A.
    mul!(A, B, W)
    A .= A .+ b
    A .= h.(A)
end

"Computes the weight gradient of a network with respect to a given data set."
function backprop!(net, data::AbstractMatrix{Float64},
        targets::AbstractMatrix{Float64})
    k, n = batchsize(net), size(data, 1)
    for i = 1:k:n
        j = min(i+k-1, n)
        feedforward!(net, @view data[i:j,:])
        feedback!(net, @view targets[i:j,:])
    end
end

"Computes and stores error derivatives, with respect to activations, for a
 network's units."
function feedback!(net, target_batch::AbstractMatrix{Float64})
    B, A = net.errors[end], net.activities[end]
    B .= A .- target_batch
    B .*= dh.(A)
    for l = layers(net)-1:-1:1
        # B contains the errors for layer l+1.
        W, G = net.weights[l+1], net.gradients[l+1]
        A, E = net.activities[l], net.errors[l]
        gradients!(G, B, A) # for the weights connecting layers l and l+1.
        errors!(E, B, A, W) # for the units in layer l.
        B = E
    end
end

"Computes error derivatives with respect to activations for a layer of units."
function errors!(E, B, A, W)
    # B contains the errors of the next layer, A the activities of the current
    # layer, and W the weights between the two layers. The resulting errors will
    # be stored in E.
    mul!(E, B, transpose(W))
    E .*= dh.(A)
end

"Computes error derivatives with respect to weights for a layer of units."
function gradients!(G, B, A)
    # B contains the errors of the next layer, and A the activities of the
    # current one. The gradient with respect to the weights between these layers
    # will be stored in G.
    mul!(G, transpose(A), B)
end