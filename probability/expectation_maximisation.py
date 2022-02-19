### Expectation Maximisation

import numpy as np
from scipy import stats
from scipy.special import logsumexp

def get_random_psd(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())

def initialize_random_params():
    params = {'phi': np.random.uniform(0, 1),
              'mu0': np.random.normal(0, 1, size=(2,)),
              'mu1': np.random.normal(0, 1, size=(2,)),
              'sigma0': get_random_psd(2),
              'sigma1': get_random_psd(2)}
    return params

def e_step(x, params):
    # p(y) -- shape (2,)
    log_p_y = np.log([1-params["phi"], params["phi"]])
    # p(x|y) -- shape (N,2)
    log_p_x_y = np.log([stats.multivariate_normal(params["mu0"], params["sigma0"]).pdf(x),
            stats.multivariate_normal(params["mu1"], params["sigma1"]).pdf(x)])
    
    # calculate log(p(x,y))
    log_p_xy = log_p_y[np.newaxis, ...] + log_p_x_y.T
    # log(\sum p(x,y))
    log_p_xy_norm = logsumexp(log_p_xy, axis=1)
    
    # So p(y|x) is now: 
    p_y_x = np.exp(log_p_xy - log_p_xy_norm[..., np.newaxis])
    
    return log_p_xy_norm, p_y_x

def m_step(x, params):
    total_count = x.shape[0]
    
    # get the heuristics of the posterior from the e-step, i.e. p(y|x)
    # and calculate the sum(y=c|x) for each class
    _, heuristics = e_step(x, params)
    heuristic0 = heuristics[:, 0]
    heuristic1 = heuristics[:, 1]
    sum_heuristic1 = np.sum(heuristic1)
    sum_heuristic0 = np.sum(heuristic0)
    
    # Calculate the new parameters
    phi = (sum_heuristic1/total_count)
    mu0 = (heuristic0[..., np.newaxis].T.dot(x)/sum_heuristic0).flatten()
    mu1 = (heuristic1[..., np.newaxis].T.dot(x)/sum_heuristic1).flatten()
    diff0 = x - mu0
    sigma0 = diff0.T.dot(diff0 * heuristic0[..., np.newaxis]) / sum_heuristic0
    diff1 = x - mu1
    sigma1 = diff1.T.dot(diff1 * heuristic1[..., np.newaxis]) / sum_heuristic1
    params = {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
    
    return params

def get_avg_log_likelihood(x, params):
    loglikelihood, _ = e_step(x, params)
    return np.mean(loglikelihood)


def run_em(x, params):
    avg_loglikelihoods = []
    while True:
        avg_loglikelihood = get_avg_log_likelihood(x, params)
        avg_loglikelihoods.append(avg_loglikelihood)
        if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
            break
        params = m_step(x, params)
        
    print("\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
               % (params['phi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']))
    _, posterior = e_step(x, params)
    forecasts = np.argmax(posterior, axis=1)
    return forecasts, posterior, avg_loglikelihoods, params

def learn_params(x_labeled, y_labeled):
    n = x_labeled.shape[0]
    phi = x_labeled[y_labeled == 1].shape[0] / n
    mu0 = np.sum(x_labeled[y_labeled == 0], axis=0) / x_labeled[y_labeled == 0].shape[0]
    mu1 = np.sum(x_labeled[y_labeled == 1], axis=0) / x_labeled[y_labeled == 1].shape[0]
    sigma0 = np.cov(x_labeled[y_labeled == 0].T, bias= True)
    sigma1 = np.cov(x_labeled[y_labeled == 1].T, bias=True)
    return {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# to make outputs reproducible across runs
np.random.seed(42)

# create data set
nSamples = 100
X = 2 * np.random.rand(nSamples,1)
y = 4 + 3 * X + np.random.randn(nSamples, 1)

# plot data set
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])


X_m = np.c_[np.ones((100,1)), X]           # add x0 = 1 to each instance
eta = 0.1
n_iteration = 100                          # a fixed number of iterations
w = np.random.randn(2,1)

# to rpedict the regression line from x=0 to 2 
X_new = np.array([[0],[2]])                 # min and max x-values
X_new_m = np.c_[np.ones((2,1)), X_new]      # add a column 0 with all 1
tr = 0.2

for iteration in range(n_iteration):
    gradients = 2/nSamples * X_m.T.dot( X_m.dot(w) - y )  
    w = w - eta * gradients

    if iteration % 25 == 0:      
        y_predict = X_new_m.dot(w)    # predicted y values for each X_new value
        plt.plot(X_new, y_predict, 'r-', alpha=tr)
        tr = min(tr + 0.1, 1.0)

y_predict = X_new_m.dot(w)    # predicted y values for each X_new value
plt.plot(X_new, y_predict, 'r-')
plt.show()
