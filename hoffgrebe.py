'''
estimates log likelihood of estimated covariance matrix
based on leave-one-out cross-validation (LOOC), as developed
by Hoffbeck and Landgrebe [1], and extended in [2].
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

#N_MINUS_NSTAR=0 ## use convention that n*=n (max likelihood)
N_MINUS_NSTAR=1 ## use convention that n*=n-1 (unbiased)


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
    c = np.trace(R)/d
    return c*np.eye(d)

def shrink(R,shrinktarget=None):
    '''shrink a covariance matrix to a target matrix
    R: ridge
    D: diag
    '''
    try:
        ## upper case first character
        shrinktarget = shrinktarget.upper()[0]
    except (TypeError,AttributeError):
        shrinktarget = None
    shrinker = shrink_diag if shrinktarget=='D' else shrink_ridge
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
    if np.any(z<0):
        v.print(f'{b=}, 1/b={1/b}, max(r)={max(r)}')
    return np.log(z) + r/z

class HoffGrebe_Base:
    '''
    used as base class for HoffGrebe class; does or sets up
    computations that don't invovle the regularization variable a.
    in particular, defines: n, nstar, mu, S
    '''
    def __init__(self,X,meansubtract=True):
        assert len(X.shape)==2
        self.n,_ = X.shape
        if meansubtract:
            self.nstar = self.n - N_MINUS_NSTAR
            self.mu,self.S = bb.im_mu_covar(X)
            self.S *= (self.n/self.nstar)
        else:
            self.nstar = self.n
            self.mu=0
            self.S = bb.im_covar(X,0)


    def T(self,shrinktarget=None):
        '''creates shrinkage target'''
        return shrink(self.S,shrinktarget=shrinktarget)

class HoffGrebe(HoffGrebe_Base):
    '''Hoffbeck-Landgrebe LOOC estimator of log likelihood'''
    def __init__(self,X,a,shrinktarget=None,meansubtract=None):
        super().__init__(X,meansubtract=meansubtract)
        T = self.T(shrinktarget=shrinktarget)

        n = self.n
        nstar = self.nstar
        if meansubtract:
            b = n*(1-a)/((n-1)*(nstar-1))
            Ga = nstar*(n-1)*b*self.S/n + a*T
        else:
            b = (1-a)/(n-1)
            Ga = n*b*self.S + a*T
        self.Ga_inv = la.inv(Ga)
        _,self.logdet_Ga = la.slogdet(Ga)

        ## first two terms of log likelihood
        d = self.S.shape[0]
        self.loglikett = (d*np.log(2*np.pi) + self.logdet_Ga)/2

        ## keep parameters as attributes to self
        self.a = a
        self.b = b

    def fmean(self,X):
        '''estimated mean value of function f()'''
        rk = bb.im_gen_dotproduct(X-self.mu, self.Ga_inv)
        return np.mean( f(self.b,rk) )

    def log_likelihood_est(self,X):
        '''estimated log likelihood, based on mean of f()'''
        return self.loglikett + self.fmean(X)/2

    def fmean_mmapprox(self):
        '''mean-Mahalanobis approximation to mean of f'''
        ro = np.trace(np.dot(self.Ga_inv,self.S))
        ro = ro*(self.nstar/self.n)
        return f(self.b,ro)

    def log_likelihood_prox(self):
        '''estimated log likelhood, based on mean Mahalnobis method'''
        return self.loglikett + self.fmean_mmapprox()/2

    def log_like_quadterm(self,X):
        '''quadratic correction term for mean Mahalanobis estimator'''
        rk = bb.im_gen_dotproduct(X-self.mu, self.Ga_inv)
        b = self.b
        ro = np.mean(rk)
        rkvar = np.var(rk)
        return (rkvar/2)*b*(b*b*ro-b+2)/((1-b*ro)**3)

    def log_likelihood_mm(self,X_mc=None):
        '''est log likelihood bease on mean Mahalanobis method,
        possibly including a quadratic correction if X_mc supplied
        '''
        loglike = self.log_likelihood_prox()
        if X_mc is not None:
            ## use X_mc to estimate quadratic term
            loglike += self.log_like_quadterm(X_mc)
        return loglike

def mc_subsample(X,nsub=0):
    '''monte carlo subsample'''
    n,d = X.shape
    if nsub == 0:
        nsub = n
    if nsub < n:
        ndx = np.random.choice(n,nsub,replace=False)
        assert len(X.shape)==2
        X_mc = X[ndx,:]
    else:
        X_mc = X
    v.vprint('mc:',d,nsub,n,X.shape,X_mc.shape)
    return X_mc

def hoffgrebe(X,a,shrinktarget=None):
    '''Hoffbeck-Landgrebe estimator of log likelihood
    Input:
    X: data matrix (or datacube)
    a: shrinkage coefficient
    shrinktarget: R or D for type of shrinkage target
    '''
    hg = HoffGrebe(X,a,shrinktarget=shrinktarget)
    return hg.log_likelihood_est(X)

def hoffgrebe_mc(X,a,shrinktarget=None,monte_carlo=None,X_mc=None):
    '''Monte-Carlo approximation to Hoffbeck-Landgrebe'''
    hg = HoffGrebe(X,a,shrinktarget=shrinktarget)
    if X_mc is None:
        X_mc = mc_subsample(X,nsub=monte_carlo)
    return hg.log_likelihood_est(X_mc)

def hoffgrebe_mm(X,a,shrinktarget=None,X_mc=None):
    '''Mean Mahalanobis approximation to Hoffbeck-Landgrebe'''
    hg = HoffGrebe(X,a,shrinktarget=shrinktarget)
    return hg.log_likelihood_mm(X_mc=X_mc)
