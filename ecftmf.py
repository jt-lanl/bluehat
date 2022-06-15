## Various routines for EC-FTMF like detectors
## Based on quantities A, B, C
## With: A = (x-mu)'Rinv (x-mu)/d    ## RX/d
##       B = (t-mu)'Rinv (x-mu)/d    ## MF/d
##       C = (t-mu)'Rinv (t-mu)/d    ##  T/d
## Note that these A,B,C values are /unrelated/ to those in IGARSS 2020 paper
## Meanwhile, M = mu'Rinv(x-mu)/d would be useful, since B-M then gives
## the standard additive matched filter.  Further B2 = (t2-mu)'Rinv... is
## a second target, and B2-M is that second target's additive matched filter

## note, we /always/ want to clip alpha to [0,1]; else, get funny results
## even though we do notice that Pfa>0.5 performance is truncated for some cases

import numpy as np
import scipy.linalg as la

import verbose as v
from . import basic,whiten
## Coordinate conversion

def mfr_to_abc(X,Y,C):
    B = Y*np.sqrt(C)
    A = X*X+B*B/C
    return A,B,C

def abc_to_mfr(A,B,C):
    x = np.sqrt(A - B*B/C)
    y = B/np.sqrt(C)
    return x,y

def matched_ABC(a,A,B,C):
    AA = (1-a)*(1-a)*A + 2*a*(1-a)*B + a*a*C
    BB = (1-a)*B + a*C
    return AA,BB,C

def daptive_alpha(B,C):
    alpha = B/C
    return np.clip(alpha,0,0.99999)

## matched filter & ACE

def amf_DABC(nu,d,A,B,C):
    ## note, this is replacement-model AMF; ie, (t-mu)'Rinv(x-mu)/d
    return B

def ace_DABC(nu,d,A,B,C):
    return B/np.sqrt(A)

def ecglrt_DABC(nu,d,A,B,C):
    if nu<2:
        return B
    else:
        return B*np.sqrt((nu-1)/(nu-2+d*A))

## Gaussian FTMF

def ftmf_alpha(BC,BA):
    alpha = 1-0.5*(BC + np.sqrt(BC*BC - 4*(BA+BC)))
    v.vprint("ftmf alpha:",np.min(alpha),np.max(alpha))
    return np.clip(alpha,0,1)

def ftmf_logpza(a,A,B,C):
    return -np.log(1-a)-0.5*(A-2*a*B+a*a*C)/((1-a)*(1-a))

def ftmf_DABC(A,B,C):
    a = ftmf_alpha(B-C,B-A)
    return ftmf_logpza(a,A,B,C) - ftmf_logpza(0,A,B,C)

def ftmf_generate(N,d,C,a=0):
    raise RuntimeError("Deprecated")
    z = np.random.normal(size=[d,N])
    u = np.ones([d,1])*np.sqrt(C)
    z = (1-a)*z + a*u
    A = np.sum(z*z,axis=0)/d
    B = np.sum(z*u,axis=0)/d
    return A,B,C

## Multivariate-t (EC) FTMF

def ecftmf_alpha(nu,d,A,B,C):
    if nu<2:
        return ftmf_alpha(B-C,B-A)
    AA= nu-2 + d*C
    BB= (d-nu)*(B-C)
    CC= -nu*(A-2*B+C)
    oma = (-BB+np.sqrt(BB*BB-4*AA*CC))/(2*AA)
    #print("oma:",np.min(oma),np.max(oma))
    return np.clip(1-oma,0,1)

def ecftmf_logpza(a,nu,d,A,B,C):
    '''actually returning (1/d)*log Pza'''
    if nu<2:
        return ftmf_logpza(a,A,B,C)
    #return -d*np.log(1-a) - 0.5*(d+nu)*np.log(1+d*(A-2*a*B+a*a*C)/((1-a)*(1-a)*(nu-2)))
    logpza = -np.log(1-a) - 0.5*(1+nu/d)*np.log((nu-2)+d*(A-2*a*B+a*a*C)/((1-a)*(1-a)))
    if 0: #nu>2:
        ## this is just a constant...does it matter?
        logpza += 0.5*(1+nu/d)*np.log(nu-2)
    return logpza

def ecftmf_DABC(nu,d,A,B,C):
    a = ecftmf_alpha(nu,d,A,B,C)
    #print("a:",np.min(a),np.max(a))
    return ecftmf_logpza(a,nu,d,A,B,C)-ecftmf_logpza(0,nu,d,A,B,C)

def ftce_DABC(d,A,B,C):
    '''ecftmf in limit as nu->2'''
    return ecftmf_DABC(2.0,d,A,B,C)

def ecftmf_daptive_DABC(nu,d,A,B,C):
    a = daptive_alpha(B,C)
    return ecftmf_logpza(a,nu,d,A,B,C)-ecftmf_logpza(0,nu,d,A,B,C)

def ecftmf_cv_DABC(a,nu,d,A,B,C):
    return ecftmf_logpza(a,nu,d,A,B,C)-ecftmf_logpza(0,nu,d,A,B,C)

def ecftmf_threesigma_DABC(nu,d,A,B,C):
    threesig = 3/np.sqrt(C*d)
    if threesig > 1:
        threesig = 0.999
    return ecftmf_cv_DABC(threesig,nu,d,A,B,C)

def ecftmf_lmp_DABC(nu,d,A,B,C):
    '''lmp = Locally Most Powerful -- related to Rao test
    corresponds to a->0 limit; equiv: (d/da)[log p(x|a)]'''
    if nu<2:
        return B-A
    else:
        return (B-A)/(1+d*A/(nu-2))


## Generate data
## nb, if N is huge and d is large, can generate in blocks
def ecftmf_generate(N,nu,d,C,a=0):
    z = np.random.normal(size=[d,N])
    if nu > 2:
        R = np.random.chisquare(nu,size=[1,N])
        R = np.sqrt((nu-2)/R)
        z = z*R
    u = np.ones([d,1])*np.sqrt(C)
    #print("u.u/d:",np.dot(u.T,u)/d)
    z = (1-a)*z + a*u
    A = np.sum(z*z,axis=0)/d
    B = np.sum(z*u,axis=0)/d
    #print("z.z/d:",np.mean(A))
    #print("z.u/d:",np.mean(B))
    #print("std(z.u/d)",np.std(B))
    return A,B,C

def ecftmf_addtarget(a,A,B,C):
    '''
    for target abundance a,
    and summarized data A,B,C, corresponding to target-free background
    return summarized At,Bt,Ct, corresponding to target abundance a
    '''
    At = (1-a)*(1-a)*A + 2*a*(1-a)*B + a*a*C
    Bt = (1-a)*B + a*C
    Ct = C
    return At,Bt,Ct

def ecftmf_getABC(xcube,tgt,spectral_axis=None):
    '''Assumes xdata is in BIP order
    Given target-free data z
    and target t
    return summarized data A,B,C
    '''
    ## Not sure we /have/ to flatten cube here,
    ## as long as we can make tgt broadcastable
    
    spectral_axis = basic.get_spectral_axis(spectral_axis)
    xdata,im_shape = basic.cubeflat(xcube,spectral_axis=spectral_axis)
    _,d = xdata.shape
    W = whiten.Whitener(xdata,e=1.0e-12,spectral_axis=spectral_axis)
    wdata = W.whiten(xdata)
    wtgt = W.whiten(tgt)

    A = np.sum(wdata**2, axis=-1)/d
    B = np.sum(wdata * wtgt.reshape(1,d), axis=-1)/d
    C = np.sum(wtgt**2, axis=-1)/d

    A = A.reshape(im_shape)
    B = B.reshape(im_shape)

    return A,B,C

## Old and obsolete?
def ecftmf_getABC_BSQ(z,t):
    '''Assumes z is in BSQ order
    Given target-free data z
    and target t
    return summarized data A,B,C
    '''
    zshape = z.shape
    z = z.reshape(zshape[0],-1)

    d,N = z.shape
    mu = np.mean(z,axis=1).reshape(-1,1)
    t = t.reshape(-1,1)
    R = (z - mu) @ (z-mu).T / N
    e = 1.0e-5  ### Hardcoded!
    e = 1.0e-12
    Rinv = la.inv((1-e)*R + e*np.diag(np.diag(R)))
    if 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4,3))
        plt.title("typical spectra")
        plt.plot(z[:,0],"c-",label="z")
        for n in range(1,N,1000):
            plt.plot(z[:,n],"c-")
        #plt.plot(mu,label="mu",lw=2,color="blue")
        tt = mu + (t-mu)/0.05
        plt.plot(tt,label="t",lw=2,color="green")
        plt.plot(t,label="t'",lw=2,color="magenta")
        plt.xlabel("Channel index")
        plt.ylabel("Reflectance")
        plt.legend()
        plt.tight_layout()
        plt.savefig("spectra.pdf")
        #plt.show()
    A = np.sum( (z-mu) * (Rinv @ (z-mu)), axis=0)/d
    B = np.sum( (t-mu) * (Rinv @ (z-mu)), axis=0)/d
    C = np.sum( (t-mu) * (Rinv @ (t-mu))        )/d

    A = A.reshape(zshape[1:])
    B = B.reshape(zshape[1:])

    return A,B,C



def ecftmf_generate_blocks(N,nu,d,C,a=0,blocks=1):
    A = np.zeros(N*blocks,dtype=float)
    B = np.zeros(N*blocks,dtype=float)
    for b in range(blocks):
        z = np.random.normal(size=[d,N])
        if nu > 2:
            R = np.random.chisquare(nu,size=[1,N])
            R = np.sqrt((nu-2)/R)
            z = z*R
        u = np.ones([d,1])*np.sqrt(C)
        #print("u.u/d:",np.dot(u.T,u)/d)
        z = (1-a)*z + a*u
        A[b*N:b*N+N] = np.sum(z*z,axis=0)/d
        B[b*N:b*N+N] = np.sum(z*u,axis=0)/d

    return A,B,C

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from bluehat import rocu

    C=4.0

    nu=3.0
    d=100
    delta=0.01
    x = np.arange(0.01, 4.0, delta)
    y = np.arange(-1.5, 2.5, delta)
    X,Y = np.meshgrid(x, y)
    A,B,C = mfr_to_abc(X,Y,C)
    D = ecftmf_DABC(nu,d,A,B,C)

    plt.figure()
    #CS = plt.contour(X, Y, D, 40, colors='k')
    CS = plt.contour(X,Y,D, levels=[0,0.005,0.01,0.1,0.2,0.3,0.4,0.5,1.0],colors='k')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('Residual')
    plt.ylabel('Matched Filter')
    plt.title('EC-FTMF Detector D(MF,R)')
    print("Range D: ",np.min(D),np.max(D))

    d=100
    N=1000
    aa,bb,cc = ecftmf_generate(N,nu,d,C,0)
    x,y = abc_to_mfr(aa,bb,cc)
    plt.plot(x,y,'.')

    DD = ecftmf_DABC(nu,d,aa,bb,cc)
    DD = np.sort(DD)
    Dmax = np.max(DD)
    D5 = DD[-5]

    DDo = ftmf_DABC(aa,bb,cc)
    DDC = ecftmf_cv_DABC(0.1,nu,d,aa,bb,cc)
    DDlmp = ecftmf_lmp_DABC(nu,d,aa,bb,cc)

    plt.contour(X,Y,D, levels=[D5,Dmax],colors='m')

    aa,bb,cc = ecftmf_generate(N,nu,d,C,0.1)
    x,y = abc_to_mfr(aa,bb,cc)
    plt.plot(x,y,'r.')

    DDx = ecftmf_DABC(nu,d,aa,bb,cc)
    DDxo = ftmf_DABC(aa,bb,cc)
    DDxC = ecftmf_cv_DABC(0.1,nu,d,aa,bb,cc)
    DDxlmp = ecftmf_lmp_DABC(nu,d,aa,bb,cc)


    fa,pd = rocu.roc(DDx,DD)
    plt.figure()
    plt.semilogx(fa,pd,label='EC-FTMF')
    fa,pd = rocu.roc(DDxo,DDo)
    plt.semilogx(fa,pd,'r',label='FTMF')
    fa,pd = rocu.roc(DDxC,DDC)
    plt.semilogx(fa,pd,'g',label='Clairvoyant')
    fa,pd = rocu.roc(DDxlmp,DDlmp)
    plt.semilogx(fa,pd,'m',label='LMP')


    plt.legend()

    plt.title("ROC")

    plt.show()
