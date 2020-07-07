#==============================================================================#
# RJNN: Forward- and back-propagation
#==============================================================================#

#==============================================================================#
# Notes:
# 1. At the time of writing, @view calls still allocate a small amount of memory
#  on the heap (and add a bit of execution time). These calls are necessary when
#  dealing with the final, partial batch of data, which is handled separately
#  from the other batches in the functions below. If views were 'free', we could
#  simply apply them to all of the batches and make the code a bit neater.
#==============================================================================#

# h(x) = 1.71590471*tanh.(0.66666667x)
# dh(h) = 1.14393647*(1 - 0.33963596h^2) # takes h(x) as an argument.
h(x) = tanh.(x)
dh(h) = (1 - h^2)

"Applies a neural network to a given set of observations."
function predict!(A::AbstractMatrix{Float64}, net,
        data::AbstractMatrix{Float64})
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
    k = size(data_batch, 1)
    partial = k < batchsize(net)
    B = partial ? @view(net.activities[1][1:k,:]) : net.activities[1]
    B .= data_batch
    for l = 1:depth(net)
        # B contains the activities for layer l-1 (stored at index l).
        W, b = net.weights[l], net.biases[l]
        A = partial ? @view(net.activities[l+1][1:k,:]) : net.activities[l+1]
        activities!(A, B, W, b)
        B = A
    end
end

"Computes activities for a single layer of units."
function activities!(A, B, W, b)
    # B contains the activities of the previous layer, and W and b the weights
    # and biases of the current layer. The resulting activities h(B*W + b) will
    # be stored in A.
    mul!(A, B, W)
    A .= A .+ b
    A .= h.(A)
end

"Computes the weight gradient of a network with respect to a given data set."
function backprop!(net, data::AbstractMatrix{Float64},
        targets::AbstractMatrix{Float64})
    # In addition to computing the weight gradient, this function computes the
    # gradient with respect to the activations and inputs as well. One could
    # easily alter it to store these derivatives in an auxiliary matrix (similar
    # to the one used in the prediction function).
    k, n = batchsize(net), size(data, 1)
    for G in net.gradients
        fill!(G, 0.)
    end
    for i = 1:k:n
        j = min(i+k-1, n)
        feedforward!(net, @view data[i:j,:]) # activities.
        feedback!(net, @view targets[i:j,:]) # errors and gradient.
    end
end

"Computes error derivatives with respect to the network's activations and uses
 these to update the weight gradient."
function feedback!(net, target_batch::AbstractMatrix{Float64})
    k = size(target_batch, 1)
    partial = k < batchsize(net)
    A, B = net.activities[end], net.errors[end]
    if partial
        A, B = @views A[1:k,:], B[1:k,:]
    end
    outerrors!(B, A, target_batch) # function barrier for type stability.
    for l = depth(net):-1:1
        # B contains the errors for layer l (stored at index l+1).
        W, G = net.weights[l], net.gradients[l]
        A, E = net.activities[l], net.errors[l]
        if partial
            A, E = @views A[1:k,:], E[1:k,:]
        end
        gradient!(G, B, A) # for the weights connecting layers l-1 and l.
        if l > 1
            errors!(E, B, A, W) # for the units in layer l-1.
        else
            inerrors!(E, B, W) # for the input units.
        end
        B = E
    end
end

outerrors!(E, A, T) = E .= (A .- T) .* dh.(A)
inerrors!(E, B, W) = mul!(E, B, transpose(W))

"Computes error derivatives with respect to the activations of a hidden layer."
function errors!(E, B, A, W)
    # B contains the errors of the next layer, A the activities of the current
    # layer, and W the weights between the two layers. The resulting errors will
    # be stored in E.
    mul!(E, B, transpose(W))
    E .*= dh.(A)
end

"Computes error derivatives with respect to the weights between two layers and
 adds these to the current gradient."
function gradient!(G, B, A)
    # B contains the errors of the next layer, and A the activities of the
    # current one. The gradient with respect to the weights between these layers
    # will be added to G (as in G += A'*B).
    mul!(G, transpose(A), B, 1., 1.)
end