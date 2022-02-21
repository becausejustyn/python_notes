
#batch_gradient_descent

import numpy as np

m = 100
X = 2*np.random.rand(m, 1)
X_b = np.c_[np.ones((m, 1)), X]
y = 4 + 3*X + np.random.rand(m, 1)

def batch_gradient_descent():
    n_iterations = 1000
    learning_rate = 0.05
    thetas = np.random.randn(2, 1)
    thetas_path = [thetas]
    for i in range(n_iterations):
        gradients = 2*X_b.T.dot(X_b.dot(thetas) - y)/m
        thetas = thetas - learning_rate*gradients
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
bgd_thetas = np.array(batch_gradient_descent())

mbgd_len = len(bgd_thetas)
print("number of iterations:", mbgd_len)