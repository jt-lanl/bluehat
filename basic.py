'''basic.py:

Basic utilities for spectral image processing:

'''

import warnings
import numpy as np
import verbose as v

# Convention for image arrays
# [nPixel,nBand]
# [nRow,nSample,nBand]  ('Row' in ENVI is called 'Line')

# Convention
# covar(X) assumes X is a 2d array
# im_covar(X) works for X 3d or 2d, but needs to know spectral axis if 3d
#             obtains this from spectral_axis keyword, or DEFAULT
# imBSQ_covar(X) equivalent to im_covar(X,spectral_axis=0)
# imBIL_covar(X) equivalent to im_covar(X,spectral_axis=1)
# imBIP_covar(X) equivalent to im_covar(X,spectral_axis=-1)

DEFAULT_SPECTRAL_AXIS = None
def set_spectral_axis(spectral_axis):
    '''
    user must use this function to set the value 
    of the global DEFAULT_SPECTRAL_AXIS
    '''
    global DEFAULT_SPECTRAL_AXIS
    spectral_axis=int(spectral_axis)
    if spectral_axis not in [-1,0,1,2]:
        raise RuntimeError(f'Invalid spectral axis={spectral_axis}')
    DEFAULT_SPECTRAL_AXIS = spectral_axis
    
def get_spectral_axis(spectral_axis=None):
    '''return a specific spectral_axis value (None->default)'''
    if spectral_axis is not None and DEFAULT_SPECTRAL_AXIS is not None:
        if spectral_axis != DEFAULT_SPECTRAL_AXIS:
            ## Case could be made for allowing keyword spectral_axis
            ## to over-ride the default spectral_axis; in which case
            ## the following RuntimeError should only be a warning
            raise RuntimeError(f'argument spectral_axis={spectral_axis} '
                               f'does not agree with default '
                               f'spectral_axis={DEFAULT_SPECTRAL_AXIS}')
    if spectral_axis is None and DEFAULT_SPECTRAL_AXIS is None:
        warnings.warn(f'must specificy argument spectral axis, since '
                      f'the default spectral axis has not been set')
    return DEFAULT_SPECTRAL_AXIS if spectral_axis is None else spectral_axis

def validate_spectral_axis(xdata,spectral_axis):
    '''
    ensure that specified spectral_axis is consistent:
    depending on DEFAULT_SPECTRAL_AXIS, and
    depending on dimension of xdata
    returns True if okay; raises RuntimeError if not okay
    '''
    if xdata.ndim == 2 and spectral_axis is not None:
        raise RuntimeError('Cannot specify spectral_axis for 2d data')
    if xdata.ndim == 3:
        spectral_axis = get_spectral_axis(spectral_axis)
        if spectral_axis is None:
            raise RuntimeError('Must specify spectral_axis for 3d data')
    return True

def ileave_to_saxis(interleave):
    '''translate interleave string to int specifying which axis is spectral'''
    if interleave.upper() == 'BSQ':
        spectral_axis = 0
    elif interleave.upper() == 'BIP':
        spectral_axis = -1 ## or 2, but I think -1 is "nicer"
    elif interleave.upper() == 'BIL':
        spectral_axis = 1
    else:
        warnings.warn(f'interleave {interleave} is not defined/supported')
        spectral_axis = None
    return spectral_axis

def cubeflat(cube,/,
             spectral_axis=None):
    '''
    Inputs:
       cube:     3d hyperspectral image cube
       spectral_axis: specifies which axis of cube is spectral
                      0 (BSQ), 1 (BIL), -1 or 2 (BIP)

    Output:
       xdata: 2d array which is just a reshape of cube [pixels,bands]
        im_shape:  dimensions of the two dimensions that were flattened (rows,samples)
    For example:
        cube is 500x400 image with 126 spectral channels
        spectral_axis=0 => BSQ cube.shape = (126,500,400)
        xdata will have shape (200000,126)
        im_shape will be (500,400)

    Notes:
        Calls to cubeflat() /should/ specify a spectral_axis, unless
           a default spectral axis has been set via set_spectral_axis()
        If input cube is a 2d array, then it will be passed through, and
           im_shape will be the one-element array [pixels]
    '''
    if cube.ndim == 2:
        ## if already flattened, then just pass it through
        ## but not if spectral_axis is set to something
        if spectral_axis is not None:
            ## should this be a warning?
            raise RuntimeError('For 2d data, do not specify keyword '
                               f'spectral_axis={spectral_axis}')
        return cube,cube.shape[:-1]

    assert cube.ndim == 3 ## make sure cube /is/ a cube
    spectral_axis = get_spectral_axis(spectral_axis)
    if spectral_axis == 0:
        ## BSQ
        d,r,s = cube.shape
        xdata = cube.reshape(d,-1).T
    elif spectral_axis in (-1,2):
        ## BIP
        r,s,d = cube.shape
        xdata = cube.reshape(-1,d)
    elif spectral_axis == 1:
        ## BIL
        r,d,s = cube.shape
        xdata = np.moveaxis(cube,1,-1).reshape(-1,d)
    elif spectral_axis is None:
        raise RuntimeError('Must specficy spectral axis')
    else:
        raise RuntimeError(f'Invalid spectral axis: {spectral_axis}')
    
    im_shape = (r,s)
    return xdata,im_shape

def cube_unflat(xdata,im_shape,/,spectral_axis=None):
    '''convert a flattened data array back into a data cube'''
    if xdata.ndim == 3:
        return xdata
    assert xdata.ndim == 2
    rs,d = xdata.shape

    if im_shape is None or len(im_shape)==1:
        ## if we are un-flattening one that was flat before being flattened...
        return xdata

    r,s = im_shape
    assert r*s == rs
    spectral_axis = get_spectral_axis(spectral_axis)
    if spectral_axis == 0:
        ## BSQ
        cube = xdata.T.reshape(d,r,s)
    elif spectral_axis in (-1,2):
        ## BIP
        cube = xdata.reshape(r,s,d)
    elif spectral_axis == 1:
        ## BIL
        cube = xdata.reshape(r,s,d)
        cube = np.moveaxis(cube,-1,1)
    return cube

def mean_spectrum(xdata):
    '''mean spectrum of a 2d data array, mean over all the pixels in the xdata'''
    assert xdata.ndim == 2
    return np.mean(xdata,axis=0)

def _get_mu_shape(spectral_axis):
    '''return a broadcastable shape for mu vector'''
    spectral_axis = get_spectral_axis(spectral_axis)
    mu_shape = [1,1,1]
    mu_shape[spectral_axis]=-1
    return mu_shape

def im_mean_spectrum(cube,/,spectral_axis=None):
    '''return a broadcastable cube'''
    ## should make sure this is consistent with other routines
    ## that might provide a separate mu for each pixel, eg the
    ## symreg package
    if cube.ndim == 2 and spectral_axis is None:
        ## if already flattened, then just pass it through
        ## but we did advertise that it should be broadcastable...
        return mean_spectrum(cube).reshape(1,-1)
    assert cube.ndim == 3
    xdata,_ = cubeflat(cube,spectral_axis=spectral_axis)
    mu = mean_spectrum(xdata)
    mu_shape = _get_mu_shape(spectral_axis)
    return mu.reshape(mu_shape)

def covar(xdata,mu=None):
    '''
    compute covariance of data array:
         C = <(x-mu)*(x-mu)'>,
    where x is pixel spectrum, and <..> is average over pixels
    if mu=None, then compute mu from data; otherwise use specified mu
    if mu=0, then compute C = <x*x'> (aka 'correlation')
    '''
    assert xdata.ndim == 2
    if mu is None:
        mu = mean_spectrum(xdata)
    ## Xmu is X minus mu
    Xmu = xdata - mu
    R = np.dot( Xmu.T, Xmu)/Xmu.shape[0]
    return R

def im_covar(cube,mu=None,/,spectral_axis=None):
    '''
    compute covariance from image cube
         C = <(x-mu)*(x-mu)'>,
    where x is pixel spectrum, and <..> is average over pixels
    if mu=None, then compute mu from data; otherwise use specified mu
    if mu=0, then compute C = <x*x'> (aka 'correlation')
    '''
    if cube.ndim == 2 and spectral_axis is None:
        ## if already flattened, then just pass it through
        return covar(cube,mu)
    assert cube.ndim == 3
    xdata,_ = cubeflat(cube,spectral_axis=spectral_axis)
    R = covar(xdata,mu)
    return R

def mu_covar(xdata):
    '''compute mean and covariance from a 2d data matrix'''
    assert xdata.ndim == 2
    mu = mean_spectrum(xdata)
    R = covar(xdata,mu)
    return mu,R

def im_mu_covar(cube,/,spectral_axis=None):
    '''compute mean and covariance from a datacube'''
    mu = im_mean_spectrum(cube,spectral_axis=spectral_axis)
    R = im_covar(cube,mu,spectral_axis=spectral_axis)
    return mu,R
    

def applyfilter(q, xdata, mu=None):
    '''
    linear filter: return q'(x-mu) for each x in xdata
    if mu is None, then compute mu as spectral mean of xdata
    mu may be scalar (scalar 0 is the only choice that makes sense)
    or mu may be a vector of shape (1,d)
    '''
    assert xdata.ndim == 2
    _,d = xdata.shape
    assert q.size == d
    if mu is None:
        mu = mean_spectrum(xdata)
    ## too much handholding in following? or should i do more?
    ## eg: if isinstance(mu,np.ndarray): mu=mu.reshape(1,d)?
    ## if mu == 0: mu = np.zeros((1,d))
    ## assert mu.size == d
    ## mu = mu.reshape(1,d)
    q = q.reshape(1,d)
    ## f.shape = (pixels,1) = (pixels,bands) x (bands,1)
    f = np.dot(xdata-mu,q.T)
    return f.reshape(-1)  ## output is 1-d array

def im_applyfilter(q,cube,mu=None,/,spectral_axis=None):
    '''
    compute q'(x-mu) for each pixel (x is pixel's spectrum)
    q and mu are d-dimensional vectors, where
    d is the number of spectral channels; and
    '''
    if cube.ndim == 2 and spectral_axis is None:
        ## if already flattened, then just pass it through
        return applyfilter(q,cube,mu)
    assert cube.ndim == 3
    xdata,im_shape = cubeflat(cube,spectral_axis=spectral_axis)
    f = applyfilter(q,xdata,mu)
    return f.reshape(im_shape)

def gen_dotproduct(xdata,Q=None,ydata=None):
    '''generalized dot product: compute x'*Q*y
    for each matched pixel pair (x,y) from xdata,ydata
    if Q is identity matrix, then this is like np.dot(x,y)
    if Q is square, then xdata and ydata should be the same shape;
    but more generally:
    xdata.shape = [pixels,x_bands]
    ydata.shape = [pixels,y_bands]
    Qmatrix.shape = [x_bands,y_bands]

    This is maybe too fancy, but:
    note: (xdata,Q,None) ==> (xdata,Q,xdata)
    note: (xdata,None,None) ==> (xdata,eye,xdata), basically squared norm of x

    Also note: this is NOT the way to do matched filters
    t'*Rinv*(x-mu), where t is a fixed vector.  Actually, it might
    even work, but it would be terribly inefficient, using an O(Nd^2)
    algorithm for an O(Nd) task.

    But it /might/ be a way to do (t-mu)'Rinv(x-mu), where mu varies over the
    image!!  It's assumed to be constant here in 'basic.py' but that's not the
    case for /all/ spectral image processing.
    '''
    if ydata is None:
        ydata = xdata
    if Q is None:
        Q = np.eye(xdata.shape[1])

    assert xdata.ndim == Q.ndim == ydata.ndim == 2 ## no images!
    assert xdata.shape[0] == ydata.shape[0] ## number of pixels
    assert xdata.shape[1] == Q.shape[0] ## number of xbands
    assert ydata.shape[1] == Q.shape[1] ## number of ybands

    Qy = np.dot(ydata,Q.T) ## has shape [pixels,x_bands]
    xQy = np.sum(xdata * Qy, axis=-1) ## has shape [pixels]
    return xQy

def im_gen_dotproduct(xcube,Q=None,ycube=None,/,spectral_axis=None):
    '''generalized dot product: compute x'*Q*y
    for each matched pixel pair (x,y) from xcube,ycube
    if Q is identity matrix, then this is like np.dot(x,y)
    if Q is square, then xcube and ycube should be the same shape;
    but more generally:
    xcube.shape = [rows,samples,x_bands]
    ycube.shape = [rows,samples,y_bands]
    Q.shape = [x_bands,y_bands]

    This is maybe too fancy, but:
    note: (xcube,Q,None) ==> (xcube,Q,xcube)
    note: (xcube,None,None) ==> (xcube,eye,xcube), basically squared norm of x
    '''
    if ycube is None:
        ycube = xcube
    if xcube.ndim == ycube.ndim == 2:
        return gen_dotproduct(xcube,Q,ycube)

    spectral_axis = get_spectral_axis(spectral_axis)
    if Q is None:
        Q = np.eye(xcube.shape[spectral_axis])

    assert Q.ndim == 2

    assert xcube.ndim == ycube.ndim == 3
    for axis in range(3):
        if axis != (spectral_axis+3)%3:
            assert xcube.shape[:axis] == ycube.shape[:axis] ## size of image
    assert xcube.shape[spectral_axis] == Q.shape[0] ## number of xbands
    assert ycube.shape[spectral_axis] == Q.shape[1] ## number of ybands

    xdata,xim_shape = cubeflat(xcube,spectral_axis=spectral_axis)
    ydata,yim_shape = cubeflat(ycube,spectral_axis=spectral_axis)
    assert xim_shape == yim_shape

    xQy = gen_dotproduct(xdata,Q,ydata)
    return xQy.reshape(xim_shape)
