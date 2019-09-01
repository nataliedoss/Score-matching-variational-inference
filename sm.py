"""
module for nonparametric score matching posterior inference
"""

import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn import mixture
from scipy.interpolate import BSpline

from lhood import *


#################################################################################

class sm_spline():
    '''
    class for score-matching estimation for exponential family using splines 
    '''
    
    def __init__(self, theta_grid, post, knots, degree):
        self.theta_grid = theta_grid
        self.post = post
        self.knots = knots
        self.degree = degree
        self.K = self.knots.shape[0] - self.degree - 1
        self.N = self.theta_grid.shape[0]
        self.coeffs = np.diag(np.repeat(1, self.K))
        self.bsMat = np.empty(shape = (self.N, self.K))
        self.dMat = np.empty(shape = (self.N, self.K))
        self.d2Mat = np.empty(shape = (self.N, self.K))
        self.k1_bar = np.zeros(shape = (self.K, ))
        self.k2_bar = np.zeros(shape = (self.K, ))
        self.A_bar = np.zeros(shape = (self.K, self.K))

        
    def estimate_unbdd(self):
        '''
        Method to estimate sufficient statistic for density on unbounded domain.

        Returns:
        Array(float, 1 x N). log of density estimate.
        '''
        
        for i in range(self.K):
            coeff_col = BSpline(self.knots, self.coeffs[i, ], self.degree)
            self.bsMat[:, i] = coeff_col(self.theta_grid, nu = 0)
            self.dMat[:, i] = coeff_col(self.theta_grid, nu = 1)
            self.d2Mat[:, i] = coeff_col(self.theta_grid, nu = 2)

        for i in range(self.N):
            theta = self.theta_grid[i]
            self.A_bar += self.post[i] * np.outer(self.dMat[i, ], self.dMat[i, ])
            self.k2_bar += self.post[i] * self.d2Mat[i, ]

        gamma_hat = np.matmul(np.linalg.inv(self.A_bar), - self.k2_bar)

        g_est = np.empty(shape = (self.N, ))
        for i in range(self.N):
            g_est[i] = np.matmul(gamma_hat.T, self.bsMat[i, ])


        return normalize_log(g_est - np.max(g_est))

    
    # Score-matching spline estimate for theta on bounded domain
    def estimate_bdd(self):

        for i in range(self.K):
            coeff_col = BSpline(self.knots, self.coeffs[i, ], self.degree)
            self.bsMat[:, i] = coeff_col(self.theta_grid, nu = 0)
            self.dMat[:, i] = coeff_col(self.theta_grid, nu = 1)
            self.d2Mat[:, i] = coeff_col(self.theta_grid, nu = 2)

        for i in range(self.N):
            theta = self.theta_grid[i]
            self.k1_bar += self.post[i] * self.dMat[i, ] * 2 * (2 * theta - 1) * theta * (1 - theta)
            self.A_bar += self.post[i] * np.outer(self.dMat[i, ], self.dMat[i, ]) * pow(theta, 2) * pow(1 - theta, 2)
            self.k2_bar += self.post[i] * self.d2Mat[i, ] * pow(theta, 2) * pow(1 - theta, 2)

        gamma_hat = np.matmul(np.linalg.inv(self.A_bar), self.k1_bar - self.k2_bar)

        # compute the solution
        g_est = np.empty(shape = (self.N, ))
        for i in range(self.N):
            g_est[i] = np.matmul(gamma_hat.T, self.bsMat[i, ])

        return normalize_log(g_est)







class sm_dirichlet():
    
    def __init__(self, theta_vec_grid, post, d, K, N):
        
        self.theta_vec_grid = theta_vec_grid
        self.post = post
        self.d = d
        self.K = K
        self.N = N


    def k1_i(self, theta_vec, i):
        vec = np.zeros(shape = (self.K, ))
        vec[i] = 1.0 / theta_vec[i]
        vec[self.K - 1] = - 1.0 / (1.0 - np.sum(theta_vec))
        return vec

    def k2_i(self, theta_vec, i):
        vec = np.zeros(shape = (self.K, ))
        vec[i] = - 1.0 / pow(theta_vec[i], 2)
        vec[self.K - 1] = - 1.0 / pow(1.0 - np.sum(theta_vec), 2)
        return vec

    def k1(self, theta_vec):
        vec = np.zeros(shape = (self.K, ))
        for i in range(self.d):
            vec += 2.0 * (2.0 * theta_vec[i] - 1.0) * theta_vec[i] * (1.0 - theta_vec[i]) * self.k1_i(theta_vec, i)
        return vec

    def k2(self, theta_vec):
        vec = np.zeros(shape = (self.K, ))
        for i in range(self.d):
            vec += pow(theta_vec[i], 2) * pow(1 - theta_vec[i], 2) * self.k2_i(theta_vec, i)
        return vec

    def A(self, theta_vec):
        mat = np.zeros(shape = (self.K, self.K))
        for i in range(self.d):
            k1i = self.k1_i(theta_vec, i)
            mat += pow(theta_vec[i], 2) * pow(1 - theta_vec[i], 2) * np.outer(k1i, k1i)
        return mat

        
    def estimate(self):
        
        k1_bar = np.zeros(shape = (self.K, ))
        k2_bar = np.zeros(shape = (self.K, ))
        A_bar = np.zeros(shape = (self.K, self.K))

        for j in range(self.N):
            theta_vec = self.theta_vec_grid[:, j]
            k1_bar += self.post[j] * self.k1(theta_vec)
            A_bar += self.post[j] * self.A(theta_vec)
            k2_bar += self.post[j] * self.k2(theta_vec)

        # Just return the parameters alpha and beta
        return np.matmul(np.linalg.inv(A_bar), k1_bar - k2_bar)
