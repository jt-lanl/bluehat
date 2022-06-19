'''Whiten multispectral data -- transform so that covariance is identity
   User can specify shrinkage(regularization) coefficient,
   and can obtain symmtric or asymmetric whitening
   (asymmetric whitening aligns axes to principal components),
   Assumes data x is already "flattened" 2d numpy array,
     Default: x.shape = (spectral,pixels)
     But specify spectral_axis as either 0 or 1 (or None*)
     if None, then will guess based on assumption that
     there are more pixels than channels
'''

import numpy as np
import verbose as v
from . import basic

def sqrtm(R):
    '''Compute matrix square root of (possibly nonsymmetric) matrix R'''
    ## for symmetric sqrt, cnoside asymmetric (eg. like invsqrtm) or
    ## Cholesky (faster, but maybe not as robust as this svd approach
    assert R.shape[0] == R.shape[1]
    U,s,V = np.linalg.svd(R)
    s = np.diag( np.sqrt(s) )
    Rsqrt = U.dot(s).dot(V)
    return Rsqrt

def invsqrtm(R,e=1.0e-6,symmetric=False,
             maxdim=0,minsv=0):
    '''Compute inverse square root of symmetric matrix R'''

    assert R.shape[0] == R.shape[1]
    ## regularize using ridge shrinkage
    U,J,_ = np.linalg.svd(R)

    ## Dimension reduction, if asked for
    ## sv_maxdim is dimension such that all singular values are > minsv*max(sv's)
    sv_maxdim = J.size
    if minsv>0:
        sv_maxdim = max(ndx for ndx,sval in enumerate(J)
                        if sval/J[0] >= minsv)
    if maxdim:
        ## Satisfy both: dim <= maxdim and sv >= minsv
        maxdim = min([maxdim,sv_maxdim])
        J = J[:maxdim]
        U = U[:,:maxdim]

    ## Sqrt, regularize, invert
    ## Regularize first so that e and minsv have same "units"
    J = (1-e)*J + e*np.sum(J)/J.size
    J = np.sqrt(J)
    J = 1.0/J
    ## asymmetric invsqrt aligns axes to PCs
    #Rinvsqrt = np.dot( np.diag(J), U.T )
    Rinvsqrt = np.dot( U, np.diag(J) )
    if symmetric:
        Rinvsqrt = np.dot( Rinvsqrt, U.T )
    return Rinvsqrt

def _validate(xdata,spectral_axis):
    '''make sure spectral_axis specification consistent with
    xdata and with the global default defined in basic.py
    '''
    return basic.validate_spectral_axis(xdata,spectral_axis)

class Whitener:
    '''
    Whitener class fits data to get mean and covariance;
    uses that to whiten subsequent data; typical usage:

    w = Whitener(x) or
    w = Whitener(x,0) or
    w = Whitener(x,xmu) or
    w = Whitener(x,e=1.0e-6,symmetric=True,spectral_axis=-1)

    xw = w.whiten(x)
    zw = w.whiten(z) ## using transfrom that was fit to x
    x = w.restore(xw) ## undoes whitening process (unwhite? color?)

    '''

    def _flat(self,xcube):
        return basic.cubeflat(xcube,spectral_axis=self.spectral_axis)
    def _unflat(self,xdata,im_shape):
        return basic.cube_unflat(xdata,im_shape,
                                 spectral_axis = self.spectral_axis)

    def __init__(self,xdata,xmu=None,/,
                 e=0,
                 symmetric=False,
                 maxdim=0,
                 minsv=0,
                 passthru=False,
                 spectral_axis = None):
        '''Whitener object is initialized by fitting to data'''
        self.passthru = passthru
        if self.passthru:
            return
        _validate(xdata,spectral_axis)
        self.spectral_axis = spectral_axis

        xdata,_ = self._flat(xdata)
        if xmu is None:
            xmu = basic.mean_spectrum(xdata)
        elif isinstance(xmu,np.ndarray):
            xmu,_ = self._flat(xmu)
        v.vprint('xmu:',xmu.shape,'range:',
                 np.min(xmu),'< xmu <',np.max(xmu))
        C = basic.covar(xdata,xmu)
        self.W = invsqrtm(C,
                          e=e,symmetric=symmetric,
                          maxdim=maxdim,minsv=minsv)
        self.Winv = None
        self.xmean = xmu

    def whiten(self,xdata):
        '''return whitened data, this includes mean subtraction'''
        if self.passthru or xdata is None:
            return xdata
        xdata,im_shape = self._flat(xdata)
        xdata = xdata - self.xmean
        wdata = xdata.dot(self.W)
        wdata = self._unflat(wdata,im_shape)
        return wdata

    def whiten_vector(self,xvector):
        '''apply whitening to single vector -- do NOT perform
        mean subtraction'''
        if self.passthru or xvector is None:
            return xvector
        xvector = xvector.reshape(-1)
        wvector = xvector.dot(self.W)
        return wvector

    ## TODO: need to test restore in test_whiten code
    def restore(self,wdata):
        '''from whitened data, return to original data coordinates'''
        if self.passthru or wdata is None:
            return wdata
        if self.Winv is None:
            if self.W.shape[0] != self.W.shape[1]:
                ## maybe use pinv in this case?
                raise RuntimeError("Cannot restore whitened data "
                                   "if dimension has been reduced")
            ## No regularization here because
            ## 1/ not needed, W be design is non-singular
            ## 2/ restoration should be as exact as possible
            self.Winv = np.linalg.inv(self.W)
        wdata,im_shape = self._flat(wdata)
        xdata = wdata.dot(self.Winv) + self.xmean
        xdata = self._unflat(xdata,im_shape)
        return xdata

    def restore_vector(self,wvector):
        '''restore whitened vactor back to its original coordinates'''
        if self.passthru or wvector is None:
            return wvector
        if self.Winv is None:
            self.Winv = np.linalg.inv(self.W)
        vecshape = wvector.shape
        wvector = wvector.reshape(-1)
        xvector = self.Winv.dot(wvector)
        return xvector.reshape(vecshape)
