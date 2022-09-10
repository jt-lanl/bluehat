'''basic (very basic!) endmember finding'''

import numpy as np
#import scipy.linalg as la
## scipy version of linalg gave error
##Intel MKL ERROR: Parameter 5 was incorrect on entry to DGESDD.
import numpy.linalg as la

def eproject(E):
    ''' from a set of k endmembers in array E,
    find the projection operator P, such that
    Px maps the point x onto the affine space defined by the endmembers
    returns xo,P so that P*(x-xo) is the affine map
    '''
    [k,d] = E.shape

    ## Set the origin at the location of the first endmember
    ## and reset other endmembers relative to that.
    xo = E[0,:]
    EE = E[1:,:] - xo
    P = np.dot(la.pinv(EE),EE) 
    return xo,P

def emnext(im,E):
    '''given image data im, and endmemebers E, compute next endmember
    x = argmax_x |Px|, where P = I-E#E with E#=pinv(E)'''

    if E is not None:

        [N,nBands] = im.shape
        [k,d] = E.shape
        assert(d == nBands)

        ## Set the origin at the location of the first endmember
        ## and reset other endmembers relative to that.
        xo = E[0,:]
        EE = E[1:,:] - xo
        P = np.eye(d) - np.dot(la.pinv(EE),EE) 

        im = np.dot( im - xo.reshape(1,-1), P )

    ndx = np.argmax( np.sum(im ** 2, axis=1) )
    #print("|Px|:",np.sum(im[ndx,:]**2))
    return im[ndx,:]

def emfind(im,K):
    ''' find K endmembers;
    here, using MaxD algorithm '''
    
    [N,d] = im.shape
    E = np.zeros((K,d),dtype=float)

    if K>d+1:
        raise RuntimeError("More endmembers than dimensions")
    
    E[0,:] = emnext(im,None)
    for k in range(1,K):
        E[k,:] = emnext(im,E[:k,:])

    return E

if __name__ == "__main__":

    N=100
    D=5
    K=6

    np.random.seed(17)
    
    img = np.random.rand(N,D)
    print("img[0]:",img[0,:])
    E = emfind(img,K)
    print("img[0]:",img[0,:]) ## make sure img didn't get overwritten
    for k in range(K):
        print("E[",k,"]:",E[k,:])


