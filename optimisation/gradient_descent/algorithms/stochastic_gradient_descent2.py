
#stochastic_gradient_descent2

import numpy as np

m = 100
X = 2*np.random.rand(m, 1)
X_b = np.c_[np.ones((m, 1)), X]
y = 4 + 3*X + np.random.rand(m, 1)

def stochastic_gradient_descent():
    n_epochs = 50
    t0, t1 = 5, 50
    thetas = np.random.randn(2, 1)
    thetas_path = [thetas]
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2*xi.T.dot(xi.dot(thetas) - yi)
            eta = learning_schedule(epoch*m + i, t0, t1)
            thetas = thetas - eta*gradients
            thetas_path.append(thetas)

    return(thetas_path)

def compute_mse(theta):
    return(np.sum((np.dot(X_b, theta) - y)**2)/m)

def learning_schedule(t, t0, t1):
    return(t0/(t+t1))

theta0, theta1 = np.meshgrid(
    np.arange(0, 5, 0.1), 
    np.arange(0, 5, 0.1)
    )

r, c = theta0.shape
cost_map = np.array([[0 for _ in range(c)] for _ in range(r)])

for i in range(r):
    for j in range(c):
        theta = np.array([theta0[i, j], theta1[i, j]])
        cost_map[i, j] = compute_mse(theta)

exact_solution = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
sgd_thetas = np.array(stochastic_gradient_descent())

sgd_len = len(sgd_thetas)
print("number of iterations:", mbgd_len)