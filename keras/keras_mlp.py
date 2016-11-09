
import numpy as np
np.random.seed(1331)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

def f(x):
    return x * 0.5 + 2


def random_up(_x):
    x = _x[0]
    return [x, f(x) + 10 * np.random.random_sample()]


def random_down(_x):
    x = _x[0]
    return [x, f(x) - 10 * np.random.random_sample()]


SIZE = 10000

X = np.random.uniform(-100, 100, SIZE).reshape(SIZE, 1)
X1 = np.asarray([random_up(k) for k in X])
X2 = np.asarray([random_down(k) for k in X])

Y1 = np.ones(SIZE).reshape(SIZE, 1)
Y2 = np.zeros(SIZE).reshape(SIZE, 1)

X_train = np.concatenate((X1, X2))
Y_train = np.concatenate((np.concatenate((Y1, Y2), axis=1), np.concatenate((Y2, Y1), axis=1)))

model = Sequential()

model.add(Dense(32, input_dim=2))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('tanh'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, Y_train, verbose=1, nb_epoch=200, batch_size=128)

