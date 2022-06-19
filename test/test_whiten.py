'''test the whiten module'''

import argparse
import numpy as np

import verbose as v
from bluehat import basic,whiten 

VERY_SMALL = 1.0e-10

def _getargs():
    '''parse options from command line'''
    argparser = argparse.ArgumentParser(description=__doc__)
    paa = argparser.add_argument
    paa("--verbose","-v",action="count",default=0,
        help="verbose")
    args = argparser.parse_args()
    return args

def biggest(x):
    return np.max(np.abs(x))

def test_mean_zero(x,spectral_axis=-1):
    '''check that max mean is << max value'''
    v.vprint(f'mean zero shape: {spectral_axis} {x.shape}')
    mx = basic.im_mean_spectrum(x,spectral_axis=spectral_axis)
    v.vprint(f'mean zero shape: {spectral_axis} {x.shape} {mx.shape}')
    v.vprint(f'mean zero means: {biggest(mx)} << {biggest(x)} ?')
    return biggest(mx)/biggest(x) < VERY_SMALL

def test_unit_covariance(x,spectral_axis=-1,tol=VERY_SMALL):
    '''check that covariance is ~1 on diagonal, <<1 off diagonal'''
    C = basic.im_covar(x,0,spectral_axis=spectral_axis)
    d = x.shape[spectral_axis]
    C_zero = C - np.eye(d)
    v.vprint("C_zero:",biggest(C_zero))
    return biggest(C_zero) < tol

def test_symmetric_matrix(W,tol=VERY_SMALL):
    return biggest(W-W.T)/biggest(W) < tol
    
def test_1(x,/,**kw):
    '''test the whitened x has mean zero and unit covariance'''

    v.vprint(f'test_1 x.shape={x.shape}')

    W = whiten.Whitener(x,**kw)
    xw = W.whiten(x)

    assert xw.shape == x.shape

    v.vprint(f'test_1 xw.shape={xw.shape}')

    assert test_mean_zero(xw,**kw)
    assert test_unit_covariance(xw,**kw)

    v.vprint('Symmetric? (may be False)',test_symmetric_matrix(W.W))

def test_2(x,**kw):
    '''
    test that whitened covariance has 
    zero mean and unit covariance;
    also that the whitening matrix is symmetric
    '''
    W = whiten.Whitener(x,e=1.0e-8,symmetric=True,**kw)
    xw = W.whiten(x)

    assert xw.shape == x.shape

    assert test_mean_zero(xw)
    assert test_unit_covariance(xw,tol=1.0e-5)
    v.vprint('Symmetric? (should be true)',test_symmetric_matrix(W.W))

def test_dimred(x,**kw):

    v.print('dimred:  x.shape:',x.shape)
    W = whiten.Whitener(x,maxdim=5,**kw)
    xw = W.whiten(x)
    v.print('dimred: xw.shape:',xw.shape)

def main(args):
    N = 20
    d = 10
    x = 1.0 + np.random.randn(N,N,d)
    Z = np.random.randn(d,d)
    x = x @ Z

    test_1(x,spectral_axis=-1)
    test_2(x,spectral_axis=-1)

    test_dimred(x,spectral_axis=-1)

    x = np.random.randn(d,N,N)
    test_1(x,spectral_axis=0)

    ## Should add a test of sqrtm for non-symmetric matrices
    

if __name__ == '__main__':

    args = _getargs()
    v.verbosity(args.verbose)
    main(args)
    
    

    
    
