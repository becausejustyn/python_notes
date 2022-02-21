#gradient_descent_base

import numpy as np
import itertools as it
import time

np.random.seed(2017)

N = 10000
sigma = 0.1
noise = sigma * np.random.randn(N)
x = np.linspace(0, 2, N)
d = 3 + 2 * x + noise
d.shape = (N, 1)

# We need to prepend a column vector of 1s to `x`.
X = np.column_stack((np.ones(N, dtype=x.dtype), x))


def py_descent(x, d, mu, N_epochs):
    N = len(x)
    f = 2 / N

    # "Empty" predictions, errors, weights, gradients.
    y = [0] * N
    w = [0, 0]
    grad = [0, 0]

    for _ in it.repeat(None, N_epochs):
        # Can't use a generator because we need to
        # access its elements twice.
        err = tuple(i - j for i, j in zip(d, y))
        grad[0] = f * sum(err)
        grad[1] = f * sum(i * j for i, j in zip(err, x))
        w = [i + mu * j for i, j in zip(w, grad)]
        y = (w[0] + w[1] * i for i in x)
    return(w)


x_list = x.tolist()
d_list = d.squeeze().tolist()  # Need 1d lists

# `mu` is a step size, or scaling factor.
mu = 0.001
N_epochs = 10000

t0 = time.time()
py_w = py_descent(x_list, d_list, mu, N_epochs)
t1 = time.time()

print(py_w)

print('Solve time: {:.2f} seconds'.format(round(t1 - t0, 2)))