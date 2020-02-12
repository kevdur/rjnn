#==============================================================================#
# RJNN: Forward- and back-propagation
#==============================================================================#

"Applies a neural network to a given set of observations."
function predict!(net, data::AbstractMatrix{Float64},
        A::AbstractMatrix{Float64})
    # A is an auxiliary matrix that will be used to store the results; it should
    # have as many rows as the data matrix and as many columns as the network's
    # output layer.
    b, n = batchsize(net), size(data, 1)
    for i = 1:b:n
        j = min(i+b-1, n) # the final batch might not be full.
        feedforward!(net, @view data[i:j,:])
        if j == i+b-1
            A[i:j,:] = net.activities[end]
        else
            A[i:n,:] = @view net.activities[end][1:n-i+1,:]
        end
    end
end

function backprop!(net, data::AbstractMatrix{Float64},
        targets::AbstractMatrix{Float64})
    nothing
end

"Computes and stores activities for all of a network's units."
function feedforward!(net, data_batch::AbstractMatrix{Float64})
    B = data_batch # data batch.
    partial = size(B, 1) < batchsize(net)
    for (W, b, A) in zip(net.weights, net.biases, net.activities)
        if partial
            A = @view A[1:size(B, 1), :]
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
    mul!(A, B, W) # activations.
    A .= A .+ b   #
    A .= 1.71590471 .* tanh.(0.66666667 .* A) # activities.
end

"Computes and stores errors for all of a network's units."
function feedback!(net, target_batch::AbstractMatrix{Float64})
    B = @view net.errors[end][size(target_batch, 1), :]
    B .-= target_batch # target errors.
    for l = 1:layers(net)

    end
end

function errors!(E, A, B)
    E, A = @views E[1:size(B, 1), :], A[1:size(B, 1), 1]
    E .= A .- B # h(x) - t.
    E .*= 1 .- A.^2 # d/dx h(x). To do: incorporate coefficients.
end