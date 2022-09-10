'''ecu.py:

Utilities for using elliptically-contoured (EC) distributions; mostly,
multivariate t distribution.  This is a generalization of the Gaussian
that has a fatter tail. Parameter nu varies from nu=2 (very fat tail)
to nu=Infinity (Gaussian).  In most of the routines, nu=0 is a
euphemism for nu=Infinity.

'''

import numpy as np
from scipy import stats

## Generic utils
import verbose as v

## Bluehat utils
from . import basic,whiten

def nu_ml_fit_1d(xdata):
    '''given 1d data xdata, used scipy to estimate nu'''
    ## no need to whiten xdata
    ## uses scipy package, which uses max likelihood
    nu,mu,scale = stats.t.fit(xdata.reshape(-1))
    return nu

def nu_ml_fit(xdata):
    '''fit nu to each channel, then average (harmonic mean)'''
    ## do not whiten (or apply PCA) to xdata
    if xdata.ndim == 1:
        xdata = xdata.reshape(-1,1)
    dim = xdata.shape[-1]
    xdata = xdata.reshape(-1,dim)
    nu_bychan = [1./nu_ml_fit_1d(xdata[:,d])
                 for d in range(dim)]
    return 1./np.mean(nu_bychan)
        

def nu_moment_estimator_fromwhite(wdata,m=1):
    '''given already-whitened data wdata, estimate nu'''
    if m==0:
        return nu_ml_fit(wdata)
    if wdata.ndim == 1:
        wdata = wdata.reshape(-1,1)
    assert wdata.ndim == 2
    _,d = wdata.shape
    r = np.sqrt( np.sum(wdata**2,axis=-1) ).reshape(-1)
    rnum = np.mean(r**(m+2))
    rden = np.mean(r**m)
    kappa = rnum/rden

    v.vvprint("nu: d,k=r3/r: ",d,kappa,"=",rnum,"/",rden,r.shape)

    if kappa <= d + m:
        est_nu = 0
    else:
        est_nu = 2 + m*kappa/(kappa-(d+m))

    return est_nu



def nu_moment_estimator(xdata,xmean=None,m=1,spectral_axis=None,**kw):
    ## output should have nu > 2 or else nu=0
    ## where nu=0 is a euphemism for nu -> infinity
    ## Perhaps should make an exceedence plot??
    '''Implements moment estimator in appendix of EC-ACD paper[*]
    Inputs: 
        xdata (2d data array or 3d image cube)
        xmean (1d mean spectrum, or 0, or None; optional, default=None)
              if None, then use computed mean from xdata
    Keywords arguments:
        m: moment to use, default = 1
        spectral_axis: if xdata is 3d, specify its spectral axis
        **kw arguments: passed to whiten.Whitener         

    [*]J. Theiler, C. Scovel, B. Wohlberg, and B. R. Foy.  "Elliptically
    contoured distributions for anomalous change detection in
    hyperspectral imagery." IEEE Geoscience and Remote Sensing Letters
    7 (2010) 271-275 doi: 10.1109/LGRS.2009.2032565.
    '''
    if xdata.ndim == 1:
        xdata = xdata.reshape(-1,1)
    if xdata.ndim == 3:
        xdata,_ = basic.cubeflat(xdata,spectral_axis=spectral_axis)
    assert xdata.ndim == 2
    W = whiten.Whitener(xdata,xmean,**kw)
    wdata = W.whiten(xdata)
    return nu_moment_estimator_fromwhite(wdata,m=m)

def nu_est_robust(xdata,mu=None,eps=1.0e-8,**kw):
    return nu_moment_estimator(xdata,xmean=mu,minsv=eps,**kw)

def ecclutter(nu,d,shape):
    ''' makes a whitened EC-distributed image of dimension d
    and pixel size given by shape (generally a 2-tuple)
    '''

    g = np.random.randn(*shape,d)
    if nu<2:
        r = np.ones(shape)
    else:
        rv = stats.chi2(nu)
        r = rv.rvs(size=shape)
        r = np.sqrt((nu-2) / r)

    r = r.reshape(*shape,1)
    return g*r


def ecmimic(cube,/,nu=None,newshape=None):
    '''
    create a new datacube that mimics the input datacube,
    but is multivariate-t with specified nu
    nu=0: Gaussian
    nu=None: estimate nu from data
    '''
    xdata,im_shape = basic.cubeflat(cube)
    _,d = xdata.shape
    if newshape is None:
        newshape = im_shape
    W = whiten.Whitener(xdata)

    if nu is None:
        ## estimate nu from data
        wdata = W.whiten(xdata)
        nu = nu_moment_estimator_fromwhite(wdata)

    wdata = ecclutter(nu,d,newshape)
    xdata = W.restore(wdata)
    xcube = basic.cube_unflat(xdata,newshape)
    return xcube

if __name__ == "__main__":
    ## Test code
    nu_true = 20
    dim     = 320
    print( "tru_nu      :",nu_true)
    im = ecclutter(nu_true,dim,[320,1444])
    estnu = nu_moment_estimator(im)
    print( "estnu (m=1):", estnu)
    estnu = nu_moment_estimator(im,m=2)
    print( "estnu (m=2):", estnu)
    im = im + 1
    estnu = nu_moment_estimator(im)
    print( "estnu (m=1):", estnu)
    estnu = nu_moment_estimator(im,m=2)
    print( "estnu (m=2):", estnu)

    im = ecmimic(im,nu=None,newshape=[320,144])
    im_mean = basic.im_mean_spectrum(im)
    print("mu:",im_mean.shape,np.min(im_mean),"< mu <",np.max(im_mean))
    estnu = nu_moment_estimator(im)
    print( "estnu (m=1):", estnu)
    estnu = nu_moment_estimator(im,m=2)
    print( "estnu (m=2):", estnu)
