'''Density models:
   MarginalProductModel: distribution is t-stat/fatexp for each component
   ECModel: distribution is radial, eg multivariate t or multivariate gauss
'''

import math
import numpy as np
from scipy.stats import norm,t

import verbose as v #pylint: disable=unused-import

from . import ecu
from . import fatexp

def t_logpdf(x,nu):
    '''log pdf of 1d t-distn where nu=0 intrpreted as 1d Gaussian distn'''
    return t.logpdf(x,nu) if nu>0 else norm.logpdf(x)

def gauss_logpdf(x):
    ''' return log of Gaussian: constant plus squared radius '''
    _,dim = x.shape
    rsq = np.sum(x**2,axis=1)
    return -0.5*rsq - (dim/2)*math.log(2*math.pi)

def multit_logpdf(x,nu):
    '''return log of pdf at each x for multivariate t distn'''
    _,dim = x.shape
    if nu <= 2:
        return gauss_logpdf(x)
    neg_log_coef = (dim/2)*math.log((nu-2)*math.pi) + \
        math.lgamma(nu/2) - math.lgamma((nu+dim)/2)
    rsq = np.sum(x**2,axis=1)
    return -neg_log_coef - 0.5*(dim+nu)*np.log(1+rsq/(nu-2))

class MarginalProductModel: #pylint: disable=no-member
    '''Base class for models of distribution based on product of marginals'''
    def __init__(self,wdata=None):
        self.dimension = None
        self.param = [] ## parameter by dimension
        if wdata is not None:
            self.fit(wdata)

    def fixparam(self,fixed_value=None,dim=0):
        '''instead of fitting model, same constant parameter for all dimensions'''
        if fixed_value is None:
            return
        if self.dimension is None:
            self.dimension = dim
        if self.dimension == 0:
            raise RuntimeError("Dimension not specified")
        self.param = np.zeros((self.dimension,))
        self.param[:] = float(fixed_value)

    def prepare_fit(self,wdata):
        '''input whitened data of shape N,d'''
        _,dim = wdata.shape
        self.dimension = dim
        self.param = np.zeros((self.dimension,))
        ## actual fit takes place in child classes

    def check_log_density(self,xdata):
        '''density as estimated by model for (whitened) data'''
        assert xdata.ndim == 2
        _,dim = xdata.shape
        assert dim == self.dimension
        ## actual log_density takes place in child classes

class MarginalStudentT(MarginalProductModel):
    '''Model distribution as product of marginal 1d t-distributions'''

    def fixparam(self,fixed_value=None,dim=0):
        if fixed_value is not None and float(fixed_value) <= 2:
            fixed_value=0
        super().fixparam(fixed_value,dim=dim)

    def fit(self,wdata):
        '''input whitened data of shape N,d'''
        self.prepare_fit(wdata)
        for d in range(self.dimension):
            #self.param[d] = ecu.nu_moment_estimator_fromwhite(wdata[:,d])
            self.param[d] = ecu.nu_ml_fit_1d(wdata[:,d])

    def log_density(self,xdata):
        '''density as estimated by model for (whitened) data'''
        self.check_log_density(xdata)
        return sum(t_logpdf(xdata[:,d],self.param[d])
                   for d in range(self.dimension))

M_DEFAULT=1
class MarginalFatExponential(MarginalProductModel):
    '''Product of fat exponentials'''
    def __init__(self,wdata=None,/,m=M_DEFAULT):
        self.m = m
        super().__init__(wdata)

    def fit(self,wdata):
        '''input whitened data of shape N,d'''
        self.prepare_fit(wdata)
        fcn_p_from_xm,_ = fatexp.m_moment_inverter(self.m)
        for d in range(self.dimension):
            self.param[d] = fatexp.p_estimate(wdata[:,d],self.m,m_inverter=fcn_p_from_xm)

    def log_density(self,xdata):
        '''density as estimated by model for (whitened) data'''
        self.check_log_density(xdata)
        return sum(fatexp.logpdf(xdata[:,d],self.param[d])
                   for d in range(self.dimension))

class ECModel:
    '''EC model'''
    def __init__(self,wdata=None):
        self.dimension = None
        self.param = None
        if wdata is not None:
            self.fit(wdata)

    def fixparam(self,fixed_nu_value=None,dim=0):
        '''instead of fitting model, same constant nu for all dimensions'''
        if fixed_nu_value is None:
            return
        if self.dimension is None:
            self.dimension = dim
        if self.dimension == 0:
            raise RuntimeError("Dimension not specified")
        self.param = float(fixed_nu_value)

    def fit(self,wdata):
        '''input whitened data of shape N,d'''
        _,dim = wdata.shape
        self.dimension = dim
        #self.param = ecu.nu_moment_estimator_fromwhite(wdata)
        self.param = ecu.nu_ml_fit(wdata)

    def log_density(self,xdata):
        '''density as estimated by model for (whitened) data'''
        assert xdata.ndim == 2
        _,dim = xdata.shape
        assert dim == self.dimension
        return multit_logpdf(xdata,self.param)

class GaussianModel(ECModel):
    '''Gaussian model is special case of EC model with nu=0'''
    def fit(self,wdata):
        _,dim = wdata.shape
        self.dimension = dim
        self.param = 0

MODELS = {
    "ec":     ECModel,
    "t":      MarginalStudentT,
    "fatexp": MarginalFatExponential,
    "g":      GaussianModel,
}

def get_model_names():
    '''return a list of names of models'''
    return list(MODELS)

def get_model_class(name):
    '''based on the name, return the appropriate density model class'''
    name = name.lower()
    if name in MODELS:
        return MODELS[name]
    ## otherwise
    raise RuntimeError(f"Invalid model name: {name}; try: {list(MODELS)}")

def get_model(name,xdata):
    '''return an initialized model object'''
    ModelClass = get_model_class(name)
    return ModelClass(xdata)
