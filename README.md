# Neural Networks in Julia (WIP)

This is a from-scratch implementation of feedforward neural networks in Julia.
There is no set goal for this work—it's really just for the sake of keeping up
with Julia's development and tinkering with neural network ideas.

Currently I've only implemented back- (and forward-) propagation, but if I get
time I'd like to look at Markov chain Monte Carlo and other Bayesian
approaches—especially those that involve altering a network's structure.

## Usage

From a Julia REPL in the project directory:

    pkg> activate .
    julia> using RJNN
    julia> using BenchmarkTools
    julia> net = RJNN.randnet(100, 10, [100, 50]) # 100 inputs, 10 outputs.
    julia> RJNN.batchsize!(net, 1000) # set the backprop batch size.
    julia> data = randn(100_000, RJNN.inputs(net)) # 100,000 data points.
    julia> targets = rand((0., 1.), 100_000, RJNN.outputs(net)) # 10 targets.
    julia> @benchmark RJNN.backprop!($net, $data, $targets)
    julia> @benchmark RJNN.predict!($targets, $net, $data) # overwrites targets.

Backpropagation and prediction for a 3-layer network like the one above take
roughly 800 ms and 600 ms respectively (on my 2015 i7 MacBook Pro).

## Performance

Attention has been paid to performance here, and at the time of writing the
backpropagation implementation is a bit slower than that of Keras _for a
very simple use case_. The code used to test this (and the accuracy of the
implementation) is in `examples/keras.jl` and `examples/keras.py`.