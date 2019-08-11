#################################################################################


import numpy as np
from sklearn import mixture
from scipy import stats
from scipy import special

from lhood import *





# Variational algorithm for Gaussian mixture with Beta prior (package implementation)
def variational_gmm_beta(x, alpha, beta):
    n = x.shape[0]
    vi = mixture.BayesianGaussianMixture(n_components = 2,
                                         covariance_type='full',
                                         weight_concentration_prior_type = 'dirichlet_distribution').fit(x)
    pi_hat = vi.predict_proba(x)[:, 0]
    S = np.sum(pi_hat)
    gamma1 = alpha + S
    gamma2 = beta + n - S
    return stats.beta.mean(gamma1, gamma2), stats.beta.var(gamma1, gamma2)


# Variational algorithm for Gaussian mixture with Beta prior (my implementation)
def variational_gmm_beta_mine(x, mu0, mu1, sigma, alpha, beta):
    n = x.shape[0]
    pi = np.zeros(shape = (n, ))

    f1 = stats.norm.pdf(x, mu1, sigma)
    f0 = stats.norm.pdf(x, mu0, sigma)

    pi[(f1 <= f0).reshape(n, )] = 0.2
    pi[(f1 > f0).reshape(n, )] = 0.8

    # set some parameters
    max_iter = 10
    threshold = .00001
    lhood_bnd = np.empty(shape = (max_iter + 1, ))
    lhood_bnd[0] = L_vi_gmm_beta(x, pi, mu0, mu1, sigma, alpha, beta)
    convergence = 1.0
    i = 0

    # Iterate
    while ( (i in range(0, max_iter)) and (abs(convergence) > threshold) ):
        S = np.sum(pi)
        gamma1 = alpha + S
        gamma2 = beta + n - S
        num = f1 * np.exp(special.digamma(gamma1))
        denom = f1 * np.exp(special.digamma(gamma1)) + f0 * np.exp(special.digamma(gamma2))
        pi = num / denom

        lhood_bnd[i + 1] = L_vi_gmm_beta(x, pi, mu0, mu1, sigma, alpha, beta)
        convergence = (lhood_bnd[i + 1] - lhood_bnd[i]) / lhood_bnd[i]
        if (convergence > 0):
            print ("Variational objective didn't increase on iteration", i)
        i += 1

    return gamma1, gamma2


