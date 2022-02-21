#gradient_descent_numpy

import numpy as np
import itertools as it
import time
import timeit

np.random.seed(2004)

N = 10000
sigma = 0.1
noise = sigma * np.random.randn(N)
x = np.linspace(0, 2, N)
d = 3 + 2 * x + noise
d.shape = (N, 1)

mu = 0.001
N_epochs = 10000

def np_descent(x, d, mu, N_epochs):
    d = d.squeeze()
    N = len(x)
    f = 2 / N

    y = np.zeros(N)
    err = np.zeros(N)
    w = np.zeros(2)
    grad = np.empty(2)

    for _ in it.repeat(None, N_epochs):
        np.subtract(d, y, out=err)
        grad[:] = f * np.sum(err), f * (err @ x)
        w = w + mu * grad
        y = w[0] + w[1] * x
    return w

np_w = np_descent(x, d, mu, N_epochs)
print(np_w)

setup = ("from __main__ import x, d, mu, N_epochs, np_descent;"
         "import numpy as np")
repeat = 5
number = 5  # Number of loops within each repeat

np_times = timeit.repeat(
    'np_descent(x, d, mu, N_epochs)', 
    setup = setup, 
    repeat = repeat, 
    number = number
    )

print(min(np_times) / number)