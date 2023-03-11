import numpy as np
from scipy.stats import truncnorm

class GaussianDataNumpy:
    def __init__(self, mu_pos, mu_neg, cov_pos=np.identity(10), cov_neg=np.identity(10), n_pos=50, n_neg=50, mean=0, std=0.2):
        '''The ratio of n_pos/n_neg and n_pos_te/n_neg_te should be the same if we have no class-prior shifts.'''
        self.n_pos_tr, self.n_neg_tr = n_pos, n_neg
        self.n_pos_te, self.n_neg_te = n_pos, n_neg
        
        self.d = len(mu_pos)
        self.mu_pos, self.mu_neg = mu_pos, mu_neg
        self.cov_pos, self.cov_neg = cov_pos, cov_neg
        self.pos_prior = self.n_pos_tr / (self.n_pos_tr + self.n_neg_tr)
        self.mean = mean
        self.std = std

    def sub_makeData(self, n_p, n_n):
        x_p = np.random.multivariate_normal(self.mu_pos, self.cov_pos, n_p)
        x_n = np.random.multivariate_normal(self.mu_neg, self.cov_neg, n_n)
        x = np.r_[x_p, x_n]
        r, r_noise = self.getR(n_p, n_n, x, self.mean, self.std)
        y = np.r_[-np.ones(n_p), np.ones(n_n)]  
        y = (y+1)//2
        return x, r, r_noise, y
        
    def makeData(self):
        x, r, r_noise, y = self.sub_makeData(self.n_pos_tr, self.n_neg_tr)
        return x.astype('float32'), r.astype('float32'), r_noise.astype('float32'), y.astype('int64')

    def getPositivePosterior(self, x):
        """Returns the positive posterior p(y=+1|x)."""
        conditional_pos = np.exp(-0.5 * (x - self.mu_pos).T.dot(np.linalg.inv(self.cov_pos)).dot(x - self.mu_pos)) / np.sqrt(np.linalg.det(self.cov_pos)*(2 * np.pi)**x.shape[0])
        conditional_neg = np.exp(-0.5 * (x - self.mu_neg).T.dot(np.linalg.inv(self.cov_neg)).dot(x - self.mu_neg)) / np.sqrt(np.linalg.det(self.cov_neg)*(2 * np.pi)**x.shape[0])
        marginal_dist = self.pos_prior * conditional_pos + (1 - self.pos_prior) * conditional_neg
        posterior_pos = conditional_pos * self.pos_prior / marginal_dist
        return posterior_pos

    def getR(self, n_pos, n_neg, x, mean, std):
        """calculating the exact positive-confidence values (r) and noisy version (r_noise). x is the input dataset"""
        r = np.zeros((n_pos+n_neg, 2)) # n times k
        r_noise = np.zeros((n_pos+n_neg, 2)) # n times k
        for i in range(len(r)):
            pos_conf = self.getPositivePosterior(x[i,:])
            r[i, 0], r[i, 1] = pos_conf, 1-pos_conf
            a, b = -min(1-pos_conf, pos_conf), min(1-pos_conf, pos_conf)
            if b < 1e-15: # truncnorm does not work with extremely small b
                noise = 0
            else:
                noise = truncnorm.rvs(a, b, loc=mean, scale=std, size=1)
            r_noise[i, 0], r_noise[i, 1] = pos_conf + noise, 1 - pos_conf - noise
        return r, r_noise
