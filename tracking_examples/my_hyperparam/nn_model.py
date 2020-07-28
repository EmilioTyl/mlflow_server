import warnings
import math

import keras
import numpy as np



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import SGD






def get_standarize(x):
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return lambda i: (i - mu) / std

def generate_model(train_x, lr, momentum):
    model = Sequential()
    model.add(Lambda(get_standarize(train_x)))
    model.add(Dense(train_x.shape[1],
                    activation='relu',
                    kernel_initializer='normal',
                    input_shape=(train_x.shape[1],)))
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer='normal'))
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer='normal'))
    model.add(Dense(1,
                    kernel_initializer='normal',
                    activation='linear'))
    model.compile(loss='mean_squared_error',
                    optimizer=SGD(
                        lr=lr,
                        momentum=momentum
                    ),
                    metrics=[])

    return model

