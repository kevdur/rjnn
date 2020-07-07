# Run `julia --project=..` from this directory and then `include("keras.jl")`.
# Before doing so though, make sure that the unscaled tanh activation function
# is uncommented in `propagation.jl` (otherwise the results will differ from
# Keras's).
using DelimitedFiles
using RJNN

net = RJNN.randnet(100, 10, [100, 50])
data = randn(100_000, RJNN.inputs(net))
targets = Matrix{Float64}(undef, size(data, 1), RJNN.outputs(net))
RJNN.predict!(targets, net, data)
for l = 1:RJNN.depth(net)
    writedlm("out/1-weights$l.tsv", net.weights[l])
    writedlm("out/1-biases$l.tsv", net.biases[l])
end
writedlm("out/1-data.tsv", data)
writedlm("out/1-targets.tsv", targets)

net = RJNN.randnet(100, 10, [100, 50])
RJNN.backprop!(net, data, targets)
for l = 1:RJNN.depth(net)
    writedlm("out/2-weights$l.tsv", net.weights[l])
    writedlm("out/2-biases$l.tsv", net.biases[l])
    writedlm("out/2-gradients$l.tsv", net.gradients[l])
end