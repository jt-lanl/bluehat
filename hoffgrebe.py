'''
estimates log likelihood of estimated covariance matrix
based on leave-one-out cross-validation (LOOC), as developed
by Hoffbeck and Landgrebe [1], and extended in [2]
[1] J. P. Hoffbeck and D. A. Landgrebe. "Covariance matrix
estimation and classification with limited training data."
IEEE Trans. Pattern Analysis and Machine Intelligence 18 (1996) 763–767.
[2] J. Theiler. "The incredible shrinking covariance estimator."
Proc. SPIE 8391 (2012) 83910P.
'''

import numpy as np
import numpy.linalg as la
import verbose as v

from . import basic as bb

def shrink_diag(R):
    '''Input:
    R: covariance matrix
    Output:
    T: diagonal-based shrinkage target diag(diag(R)
    '''
    return np.diag(np.diag(R))

def shrink_ridge(R):
    '''Input:
    R: covariance matrix
    Output:
    T: ridge-based shrinkage target c*I
       with trace(T) = trace(R)
    '''
    d,_ = R.shape
    c = np.sum(np.diag(R))/d
    return c*np.eye(d)

def shrink(R,shrinker=shrink_diag):
    '''shrink a covariance matrix to a target matrix'''
    return shrinker(R)


def true_log_likelihood(Ra,Rtrue):
    '''return the true log likelihood 
    associated with estimated covariance Ra, 
    based on true covariance Rtrue
    '''
    d,_ = Rtrue.shape
    trace = np.trace(np.dot(la.inv(Ra),Rtrue))
    _,logdet_Ra = la.slogdet(Ra)
    L_true = (d/2)*np.log(2*np.pi) + logdet_Ra + trace
    return L_true


def f(b,r):
    '''function f defined in eq(25) of SPIE 8391 paper'''
    z = 1 - b*r
    return np.log(z) + r/z

class HoffGrebe:
    '''Hoffbeck-Landgrebe LOOC estimator of log likelihood'''
    def __init__(self,X,a,shrinker=shrink_diag):
        assert len(X.shape)==2
        n,self.d = X.shape
        self.mu,self.S = bb.im_mu_covar(X)
        T = shrinker(self.S)

        self.a = a
        self.b = (1-a)/(n-1)
        Ga = n*self.b*self.S + a*T
        self.Ga_inv = la.inv(Ga)
        _,self.logdet_Ga = la.slogdet(Ga)

        ## first two terms of log likelihood
        self.loglikett = (self.d/2)*np.log(2*np.pi) + self.logdet_Ga

    def fmean(self,X):
        '''estimated mean value of function f()'''
        rk = bb.im_gen_dotproduct(X-self.mu, self.Ga_inv)
        return np.mean( f(self.b,rk) )

    def log_likelihood_est(self,X):
        '''estimated log likelihood, based on mean of f()'''
        return self.loglikett + self.fmean(X)

    def fmean_mmapprox(self):
        '''mean-Mahalanobis approximation to mean of f'''
        ro = np.trace(np.dot(self.Ga_inv,self.S))
        return f(self.b,ro)

    def log_likelihood_prox(self):
        '''estimated log likelhood, based on mean Mahalnobis method'''
        return self.loglikett + self.fmean_mmapprox()

    def log_like_quadterm(self,X):
        '''quadratic correction term for mean Mahalanobis estimator'''
        rk = bb.im_gen_dotproduct(X-self.mu, self.Ga_inv)
        b = self.b
        ro = np.mean(rk)
        rkvar = np.var(rk)
        return (rkvar/2)*b*(b*b*ro-b+2)/((1-b*ro)**3)

def mc_subsample(X,nsub=None):
    '''monte carlo subsample'''
    n,d = X.shape
    if nsub is None:
        nsub = 5*d
    if d < nsub < n:
        ndx = np.random.choice(n,nsub,replace=False)
        assert len(X.shape)==2
        X_mc = X[ndx,:]
    else:
        X_mc = X
    v.vprint('mc:',d,nsub,n,X.shape,X_mc.shape)
    return X_mc

def hoffgrebe(X,a,shrinker=shrink_diag):
    '''Hoffbeck-Landgrebe estimator of log likelihood
    Input:
    X: data matrix (or datacube)
    a: shrinkage coefficient
    shrinker: function that provides shrinkage target T
    '''
    hg = HoffGrebe(X,a,shrinker=shrinker)
    return hg.log_likelihood_est(X)

def hoffgrebe_mc(X,a,shrinker=shrink_diag,monte_carlo=None,X_mc=None):
    '''Monte-Carlo approximation to Hoffbeck-Landgrebe'''
    hg = HoffGrebe(X,a,shrinker=shrinker)
    if X_mc is None:
        X_mc = mc_subsample(X,nsub=monte_carlo)
    return hg.log_likelihood_est(X_mc)

def hoffgrebe_mm(X,a,shrinker=shrink_diag,X_mc=None):
    '''Mean Mahalanobis approximation to Hoffbeck-Landgrebe'''
    hg = HoffGrebe(X,a,shrinker=shrinker)
    L_MM = hg.log_likelihood_prox()
    if X_mc is not None:
        ## use X_mc to estimate quadratic term
        L_MM += hg.log_like_quadterm(X_mc)
    return L_MM
