
#mini_batch_gradient_descent

import numpy as np

m = 100
X = 2*np.random.rand(m, 1)
X_b = np.c_[np.ones((m, 1)), X]
y = 4 + 3*X + np.random.rand(m, 1)

def mini_batch_gradient_descent():
    n_iterations = 50
    minibatch_size = 20
    t0, t1 = 200, 1000
    thetas = np.random.randn(2, 1)
    thetas_path = [thetas]
    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            gradients = 2*xi.T.dot(xi.dot(thetas) - yi)/minibatch_size
            eta = learning_schedule(t, t0, t1)
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
mbgd_thetas = np.array(mini_batch_gradient_descent())

mbgd_len = len(mbgd_thetas)

print("Number of iterations:", mbgd_len)