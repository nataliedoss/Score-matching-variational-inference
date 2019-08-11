
from lhood import *
from vi import *
from sm import *




#################################################################################
# Gaussian mixture likelihood and Beta prior on weights
# Compare VI and SM

theta_grid = np.arange(0.01, .99, 0.01)
n = 100
alpha = 2
beta = 3
mu0 = -2.0
mu1 = 2.0
sigma = 1
d = 1
K = 2
# for now:
theta_vec_grid = np.arange(0.01, 0.99, .01)
theta_vec_grid = theta_vec_grid.reshape(d, theta_vec_grid.shape[0])
N = theta_vec_grid.shape[1]

# generate data 
x = np.random.normal(0.0, sigma, n) # the model is incorrect
x = data_gen_gmm(alpha, beta, mu0, mu1, sigma, n) # OR the model is correct



# plot the data to check it
#plt.hist(x)
#plt.suptitle("Histogram of data")
#plt.show()

# for score-matching and for the true posterior, we need the log joint
log_joint = log_joint_gmm_beta(x, theta_grid, mu0, mu1, sigma, alpha, beta)
joint = np.exp(log_joint)

# score-matching beta
tmp = sm_dirichlet(theta_vec_grid, joint, d, K, N)
sm_beta = tmp.estimate()

# variational inference
vi = variational_gmm_beta_mine(x, mu0, mu1, sigma, alpha, beta)

# true posterior
post_normalized = normalize_log(log_joint)

# plot on the log scale
#plt.plot(theta_grid, stats.beta.logpdf(theta_grid, sm_beta[0] + 1, sm_beta[1] + 1),
#         color = "orange", label = "score-matching beta")
#plt.plot(theta_grid, stats.beta.logpdf(theta_grid, vi[0], vi[1]),
#         color = "purple", label = "variational")
#plt.plot(theta_grid, safe_log(post_normalized),
#         color = "black", label = "true")
#plt.legend(loc = "lower right")
#plt.suptitle("Log posterior estimates: Gaussian mixture likelihood with Beta prior on weights")
#plt.show()


# compare some parameters
mean_true = np.sum(theta_grid * post_normalized)
var_true = np.sum(pow(theta_grid - mean_true, 2) * post_normalized)
mean_vi = stats.beta.mean(vi[0], vi[1])
var_vi = stats.beta.var(vi[0], vi[1])
mean_sm_beta = stats.beta.mean(sm_beta[0], sm_beta[1])
var_sm_beta = stats.beta.var(sm_beta[0], sm_beta[1])
print(mean_true, var_true, mean_vi, var_vi, mean_sm_beta, var_sm_beta)



#################################################################################
#################################################################################
#################################################################################
# Binomial(theta) likelihood and Beta mixture prior on theta
# Compare VI and SM 

# Parameters
alpha0 = .3
beta0 = .3
alpha1 = 3
beta1 = 3
x = 1
theta = 0.5
knots = np.arange(.1, 1, 1.0/16)
degree = 3

# the joint is needed for sm calculations and for the true posterior
logjoint_bin_beta_mix = log_joint_bin_beta_mix(x, theta_grid, alpha0, beta0, alpha1, beta1)
joint_bin_beta_mix = np.exp(logjoint_bin_beta_mix)

# score-matching spline
tmp = sm_spline(theta_grid, joint_bin_beta_mix, knots, degree)
post_sm_normalized = tmp.estimate_bdd()

# the variational algorithm in this case is a simple update
alpha_vi = np.sum(x) + (alpha0 + alpha1 - 2.0) / 2.0
beta_vi = np.sum(1.0 - x) + (beta0 + beta1 - 2.0) / 2.0

# true posterior
post_normalized = normalize_log(logjoint_bin_beta_mix)


# plot the true posterior and the estimates
#plt.plot(theta_grid, np.log(post_sm_normalized), color = "orange", label = "score-matching spline")
#plt.plot(theta_grid, stats.beta.logpdf(theta_grid, alpha_vi, beta_vi), color = "purple", label = "variational")
#plt.plot(theta_grid, np.log(post_normalized), color = "black", label = "true")
#plt.legend(loc = "lower right")
#plt.suptitle("Log posterior estimates: binomial(theta) likelihood and Beta mixture prior on theta")
#plt.show()



#################################################################################
#################################################################################
#################################################################################
# Gaussian mixture likelihood and Gaussian mixture prior on means
# Compare VI and SM


mu0 = -4.0
mu1 = 4.0
sigma_prior = 1.0
sigma = 1.0
x = 0.0
knots = np.arange(-5.0, 5.0, 1.0/20)
degree = 3
theta_grid = np.arange(-6, 6, .01) 

# the joint is necessary for the score-matching estimate and for the true posterior
log_joint_gaussian_gmm_vals = log_joint_gaussian_gmm(x, theta_grid, mu0, mu1, sigma_prior, sigma)
joint_gaussian_gmm = np.exp(log_joint_gaussian_gmm_vals)
post = joint_gaussian_gmm

# score-matching
tmp = sm_spline(theta_grid, joint_gaussian_gmm, knots, degree)
post_sm_normalized = tmp.estimate_unbdd()

# variational estimate
# this assumes the variance of the prior and likelihood are the same
mean_vi = np.mean(x) + (mu0 + mu1) / 2.0
sd_vi = np.sqrt(pow(sigma, 2.0) / 2.0)

# true posterior
post_normalized = normalize_log(log_joint_gaussian_gmm_vals)

# plot the true posterior and the estimates
plt.plot(theta_grid, safe_log(post_sm_normalized), color = "orange", label = "score-matching spline")
plt.plot(theta_grid, stats.norm.logpdf(theta_grid, mean_vi, sd_vi), color = "purple", label = "variational")
plt.plot(theta_grid, safe_log(post_normalized), color = "black", label = "true")
plt.legend(loc = "lower right")
plt.suptitle("Log posterior estimates: Gaussian likelihood and Gaussian mixture prior on means")
plt.show()



