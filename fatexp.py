'''Routines for fat exponentials'''

import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve
from scipy import interpolate
from functools import cache
import verbose as v

def c_coef(p,a):
    '''c as a function of p,a based on integral of p(x)=1'''
    return a*p/(2*gamma(1/p))

def a_param(p):
    '''a as a function of p, based on <x^2>=1'''
    return np.sqrt(gamma(3/p)/gamma(1/p))

def m_moment(m,p):
    '''what is <x^m> for a given value of p'''
    if p<0:
        p=0.01
    xm = np.sqrt(gamma(1/p)/gamma(3/p))**m
    xm = xm * gamma((m+1)/p)/gamma(1/p)
    return xm

def logpdf(x,p,c=None,a=None):
    if a is None:
        a = a_param(p)
    if c is None:
        c = c_coef(p,a)
    return np.log(c) - np.abs(a*x)**p

def fatexp(x,p,c=None,a=None):
    '''fat exponential c*exp(-|ax|^p)'''
    return np.exp(logpdf(x,p,c,a))

def m_moment_inverter_ok(m,prange=None):
    '''return a function f such that p=f(<x^m>), 
    where <x^m> is w.r.t. fatexp(x,p); this is
    the inverse of m_moment(m,p)
    '''
    if prange is None:
        prange = (0.1,6)
    pvals = np.linspace(prange[0],prange[1],10)
    xmvals = [m_moment(m,pval) for pval in pvals]
    v.vprint(' pvals:',min(pvals),max(pvals))
    v.vprint('xmvals:',min(xmvals),max(xmvals))
    v.vprint('prange:',prange)
    fcn = interpolate.interp1d(xmvals,pvals,bounds_error=False,
                               fill_value=tuple(prange))
    xmrange = (min(xmvals),max(xmvals))
    return fcn,xmrange

#@cache
def m_moment_inverter(m,prange=None):
    '''return a function f such that p=f(<x^m>), 
    where <x^m> is w.r.t. fatexp(x,p); this is
    the inverse of m_moment(m,p)
    '''
    if prange is None:
        prange = (0.1,16)
    pvals = np.exp(np.linspace(np.log(prange[0]),np.log(prange[1]),10))
    xmvals = [m_moment(m,pval) for pval in pvals]
    xmrange = (min(xmvals),max(xmvals))
    v.vprint(' pvals:',min(pvals),max(pvals))
    v.vprint('xmvals:',min(xmvals),max(xmvals))
    v.vprint('prange:',prange)
    pvals = np.log(pvals)
    xmvals = np.log(xmvals)
    prange = tuple([np.log(p) for p in prange])
    fcn = interpolate.interp1d(xmvals,pvals,bounds_error=False,
                               fill_value=tuple(prange))
    def trufcn(xm):
        xm = np.log(xm)
        pvals = fcn(xm)
        return np.exp(pvals)
    
    return trufcn,xmrange

def p_estimate_fsolve(xdata,m):
    ''' 
    use scipy.optimize.fsolve to estimate p 
    so that data moment: xm=|xdata|^m
    equals theoretical moment: m_moment(m,p)
    '''
    ## Note: this is designed for xdata from a single channel
    ## (data channel or PCA channel), but it "works" for data
    ## from multiple channels; it just treats all the data as
    ## if it were a single list of scalar values.  That's probably
    ## okay for whitened data, but maybe not so much for real
    ## data in which each channel is scaled in its own way
    xdata = xdata - np.mean(xdata)
    xdata = xdata / np.std(xdata)
    xm = np.mean(np.abs(xdata)**m)
    v.vprint(f'{xm=}')
    fcn = lambda p: m_moment(m,p)-xm
    po = fsolve(fcn,0.1) ## starting point low avoids getting stuck
    v.vprint(f'{po[0]=}')
    return po[0]
    

def p_estimate(xdata,m,m_inverter=None):
    '''Given data, provide an estimate of p'''
    if m_inverter is None:
        m_inverter,_ = m_moment_inverter(m)
    xdata = xdata.reshape(-1)
    xm = np.mean(np.abs(xdata)**m)
    v.vvprint(f'{xm=}')
    xdata = xdata - np.mean(xdata)
    xdata = xdata / np.std(xdata)
    xm = np.mean(np.abs(xdata)**m)
    v.vvprint(f'{xm=}')
    return m_inverter(xm)
    
