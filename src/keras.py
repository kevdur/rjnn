#%%
import numpy as np
from keras.initializers import RandomNormal
from keras.layers import Dense
from keras.models import Sequential

#%%
args = dict(
    activation='tanh',
    kernel_initializer=RandomNormal(0, 1),
    bias_initializer=RandomNormal(0, 1)
)
model = Sequential()
model.add(Dense(units=100, input_dim=100, **args))
model.add(Dense(units=50, **args))
model.add(Dense(units=10, **args))

#%% Load weights, data, and results from file.
weights = []
for i in range(3):
    W = np.loadtxt(f'weights{i+1}.tsv')
    weights.append(W[:-1,:]) # weights.
    weights.append(W[-1,:]) # biases.
model.set_weights(weights)

data = np.loadtxt('data.tsv')
result = np.loadtxt('result.tsv')

#%%
%%time
# data = np.random.normal(size=(100_000, 100))
R = model.predict(data)

#%% Find the largest difference between the stored and computed results. (Make
# sure the activation functions match.)
i = np.argmax(np.abs(R - result))
i, j = np.unravel_index(i, R.shape)
R[i,j], result[i,j]

# %%
