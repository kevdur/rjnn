#%% This code is written to be run using a Jupyter/IPython kernel, ideally from
# within Visual Studio Code.
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential

#%% Define the network's structure.
args = dict(
    activation='tanh',
    kernel_initializer=RandomNormal(0, 1),
    bias_initializer=RandomNormal(0, 1),
    dtype='float64'
)
inputs, outputs = 100, 10
hidden = [100, 50]
model = Sequential()
model.add(Dense(units=hidden[0], input_dim=inputs, **args))
for n in hidden[1:]:
    model.add(Dense(units=n, **args))
model.add(Dense(units=outputs, **args))
nlayers = len(hidden) + 1

#%% Load weights, data, and results from file. (Again, make sure that the
# unnormalised tanh activation function was used in the Julia run.)
def load_weights(prefix):
    weights = []
    for i in range(nlayers):
        W = np.loadtxt(f'examples/out/{prefix}weights{i+1}.tsv', ndmin=2)
        b = np.loadtxt(f'examples/out/{prefix}biases{i+1}.tsv', ndmin=1)
        weights.append(W)
        weights.append(b)
    model.set_weights(weights)

load_weights('1-')
data = np.loadtxt('examples/out/1-data.tsv', ndmin=2)
targets = np.loadtxt('examples/out/1-targets.tsv', ndmin=2)

#%% Check that the stored (Julia) and computed (Keras) results match.
T = model.predict(data)
i = np.argmax(np.abs(T - targets))
i, j = np.unravel_index(i, T.shape)
print('Max. target difference:', T[i,j] - targets[i,j])

#%% Time the prediction method.
%timeit model.predict(data)

#%% Load the second set of weights and compute the gradient with respect to
# them. 
load_weights('2-')
mse = MeanSquaredError(reduction='sum')
with tf.GradientTape() as tape:
    T = model(data)
    loss = mse(targets, T)
Gs = tape.gradient(loss, model.trainable_variables)

gradients = []
for i in range(nlayers):
    # I've used half the sum of squared errors, so we need a scaling factor to
    # compare with Keras's mean squared error.
    w_grad = 2/outputs*np.loadtxt(f'examples/out/2-gradients{i+1}.tsv', ndmin=2)
    # b_grad = np.loadtxt(f'examples/out/2-b_gradient{i+1}.tsv')
    gradients.append(w_grad)
    # gradients.append(b_grad) # not yet implemented.

for k in range(nlayers):
    GW, Gb = Gs[2*k:2*(k+1)]
    w_grad = gradients[k]
    # w_grad, b_grad = gradients[2*k:2*(k+1)]
    i = np.argmax(np.abs(GW - w_grad))
    i, j = np.unravel_index(i, GW.shape)
    print(f'{k}: Max. gradient difference (weights):', GW[i,j] - w_grad[i,j])
    # i = np.argmax(np.abs(GW - w_grad))
    # i, j = np.unravel_index(i, T.shape)
    # print(f'{k}: Max. gradient difference (biases):', Gb[i,j] - b_grad[i,j])

#%% Time backprop. (I'm not sure if this is the best way to do it.)
%%timeit
with tf.GradientTape() as tape:
    T = model(data)
    loss = mse(targets, T)
tape.gradient(loss, model.trainable_variables)

# %%
