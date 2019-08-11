#################################################################################


import numpy as np
import random
from scipy import stats
from scipy import stats
from scipy import special




#################################################################################
#################################################################################
# Functions for various priors, likelihoods, and posteriors

# safe log
def safe_log(x):
    return np.where(x > 0.0, np.log(x), -1000.0)

def normalize_log(vec):
    num = np.exp(vec)
    return (num / np.sum(num))


# Create a sample of gaussian mixture data
def data_gen_gmm(alpha, beta, mu0, mu1, sigma, n):
    theta = np.random.beta(alpha, beta, 1)
    z = np.random.binomial(1, theta, n).reshape(n, 1)
    x = np.empty(shape = (n, 1))
    x[z == 0] = np.random.normal(mu0, sigma, np.sum(z == 0))
    x[z == 1] = np.random.normal(mu1, sigma, np.sum(z == 1))
    return x

# variational objective for gaussian mixture
def L_vi_gmm_beta(x, pi, mu0, mu1, sigma, alpha, beta):
    n = x.shape[0]
    S = np.sum(pi)
    denom = safe_log( special.beta(alpha, beta) )
    num = safe_log( special.beta(alpha + S, beta + n - S) )
    main = np.sum(pi * (stats.norm.logpdf(x, mu1, sigma) + safe_log(pi)) +
                  (1 - pi) * (stats.norm.logpdf(x, mu0, sigma) + safe_log(1 - pi)))

    return (num + main - denom)




# log likelihood for gaussian mixture
def log_lk_gmm(x, theta, mu0, mu1, sigma):
    term1 = theta * stats.norm.pdf(x, mu1, sigma)
    term2 = (1 - theta) * stats.norm.pdf(x, mu0, sigma)
    log_p_x_theta = np.sum(np.log(term1 + term2))
    return log_p_x_theta

# log likelihood values for gaussian mixture
log_lk_gmm_vals = np.vectorize(log_lk_gmm, excluded = (0, 2, 3, 4))


# log joint for gaussian mixture with beta prior
def log_joint_gmm_beta(x, theta, mu0, mu1, sigma, alpha, beta):
    log_lk = log_lk_gmm_vals(x, theta, mu0, mu1, sigma)
    log_prior = stats.beta.logpdf(theta, alpha, beta)
    return log_lk + log_prior

# beta-mixture prior
def log_prior_beta_mix(theta, alpha0, beta0, alpha1, beta1):
    return np.log(0.5 * stats.beta.pdf(theta, alpha0, beta0) + 0.5 * stats.beta.pdf(theta, alpha1, beta1))

# binomial likelihood
def log_lk_binomial(x, theta):
    return np.sum(stats.binom.logpmf(x, 1, theta))

# binomial likelihood over grid of theta's
def log_lk_binomial_vals(x, theta_grid):
    N = theta_grid.shape[0]
    ret = np.empty(shape = (N, ))
    for i in range(N):
        ret[i] = log_lk_binomial(x, theta_grid[i])
    return ret

# log joint for binomial likelihood with beta-mixture prior
def log_joint_bin_beta_mix(x, theta_grid, alpha0, beta0, alpha1, beta1):   
  return log_lk_binomial_vals(x, theta_grid) + log_prior_beta_mix(theta_grid, alpha0, beta0, alpha1, beta1)


# the prior on the means is a mixture of Gaussians
def log_prior_gmm(theta, mu0, mu1, sigma_prior):
    return np.log(0.5 * stats.norm.pdf(theta, mu0, sigma_prior) + 0.5 * stats.norm.pdf(theta, mu1, sigma_prior))

# the likelihood is gaussian
def log_lk_gaussian(x, theta, sigma):
    return stats.norm.logpdf(x, theta, sigma)

# the posterior
def log_joint_gaussian_gmm(x, theta, mu0, mu1, sigma_prior, sigma):
    return log_prior_gmm(theta, mu0, mu1, sigma_prior) + log_lk_gaussian(x, theta, sigma)
