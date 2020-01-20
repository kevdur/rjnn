#==============================================================================#
# RJNN: Neural Networks
#==============================================================================#

"Immutable neural network, containing both architecture and parameters."
struct Net
    n_layers::Int # input, hidden, and output.
    n_nodes::Vector{Int} # number of units in each layer.
    weights::Vector{AbstractMatrix{Float64}} # weights[l] connects layers l and
    biases::Vector{Float64}                  # l+1.
end

"Applies a neural network to a given data set."
function predict(net::Net)
    nothing
end