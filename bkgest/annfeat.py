'''create feature vector from annulus values'''

from collections import defaultdict, Counter
import numpy as np

def SigmaDelta(A,B):
    return (A+B)/2.0, np.abs(A-B)/2.0

def nullfeatures(A,c):
    '''
    Image A, windows diameter 2*c+1, offset ignored
    '''
    nr,nc = A.shape
    return 0*A[c:nr-c,c:nc-c]

def annfeatures(A,c,i,j):
    '''
    Input image A, size of window 2*c+1, offset i,j
    Return a tuple of features corresponding to the untransformed
    values of pixels at i,j; i,-j; j,-i; etc.
    Note that the feature images are smaller than the image A
    by 2*c in each direction; eg if A is a 50x50 image and c=5,
    then features will be 40x40 images
    '''
    ## should we assert that abs(i) <= c, abs(j) <= c ???
    nr,nc = A.shape
    def Ac(ii,jj):
        return A[c+ii:nr-c+ii,c+jj:nc-c+jj]

    if i == j == 0:
        f = Ac(0,0)
        return (f,)

    if (i==0 or j==0):
        k=i+j
        return Ac(0,k),Ac(0,-k),Ac(k,0),Ac(-k,0)

    if i==j:
        return Ac(i, i), Ac(-i, i), Ac(i,-i), Ac(-i,-i)

    #else:
    return Ac(i, j),Ac(-i, j),Ac(i,-j),Ac(-i,-j),\
        Ac(j, i),Ac(-j, i),Ac(j,-i),Ac(-j,-i)


def symsigfeatures(A,c,i,j):
    nr,nc = A.shape
    def Ac(ii,jj):
        return A[c+ii:nr-c+ii,c+jj:nc-c+jj]

    f = (Ac(i,j)+Ac(-i,j)+Ac(i,-j)+Ac(-i,-j)+
         Ac(j,i)+Ac(-j,i)+Ac(j,-i)+Ac(-j,-i))/8.

    return (f,)

def ringfeatures(A,c,i):
    nr,nc = A.shape
    def Ac(ii,jj):
        return A[c+ii:nr-c+ii,c+jj:nc-c+jj]

    fsum=0
    fcount=0
    for j in range(-i,i+1):
        fsum += Ac(i,j) + Ac(-i,j)
        fcount += 2
    for j in range(-i+1,i):
        fsum += Ac(j,i) + Ac(j,-i)
        fcount += 2
    return (fsum/fcount,)

def diamondringfeatures(A,ro,ri,i):
    nr,nc = A.shape
    c=ro
    def Ac(ii,jj):
        return A[c+ii:nr-c+ii,c+jj:nc-c+jj]

    fsum=0
    fcount=0
    for j in range(i+1):
        k = i-j
        if j > ro  or k > ro:
            ## outside the annulus
            continue
        if j < ri and k < ri:
            ## inside the guard ring
            continue
        if j==0:
            fsum += Ac(0,k) + Ac(0,-k)
            fcount += 2
        elif k==0:
            fsum += Ac(j,0) + Ac(-j,0)
            fcount += 2
        else:
            fsum += Ac(j,k) + Ac(-j,k) + Ac(j,-k) + Ac(-j,-k)
            fcount += 4
    return (fsum/fcount,)

def symfeatures(A,c,i,j):
    nr,nc = A.shape
    def Ac(ii,jj):
        return A[c+ii:nr-c+ii,c+jj:nc-c+jj]

    def SigDel2(ij0,ij1):
        return SigmaDelta( Ac(*ij0), Ac(*ij1) )

    def SigDel4(ij0,ij1,ij2,ij3):
        f0,f1 = SigDel2(ij0,ij1)
        f2,f3 = SigDel2(ij2,ij3)
        f0,f2 = SigmaDelta(f0,f2)
        f1,f3 = SigmaDelta(f1,f3)
        return f0,f1,f2,f3


    if i == j == 0:
        return (Ac(0,0),)

    if (i==0 or j==0):
        k=i+j
        return SigDel4((0,k),(0,-k),
                       (k,0),(-k,0))

    if i==j:
        return SigDel4((i, i),(-i, i),
                       (i,-i),(-i,-i))

    #else:
    f0,f1,f2,f3 = SigDel4((i, j),(-i, j),
                          (i,-j),(-i,-j))
    f4,f5,f6,f7 = SigDel4((j, i),(-j, i),
                          (j,-i),(-j,-i))
    f0,f4 = SigmaDelta(f0,f4)
    f1,f5 = SigmaDelta(f1,f5)
    f2,f6 = SigmaDelta(f2,f6)
    f3,f7 = SigmaDelta(f3,f7)

    return f0,f1,f2,f3,f4,f5,f6,f7

def kleinfeatures(A,c,i,j):
    nr,nc = A.shape
    def Ac(ii,jj):
        return A[c+ii:nr-c+ii,c+jj:nc-c+jj]

    def SigDel2(ij0,ij1):
        return SigmaDelta( Ac(*ij0), Ac(*ij1) )

    def SigDel4(ij0,ij1,ij2,ij3):
        f0,f1 = SigDel2(ij0,ij1)
        f2,f3 = SigDel2(ij2,ij3)
        f0,f2 = SigmaDelta(f0,f2)
        f1,f3 = SigmaDelta(f1,f3)
        return f0,f1,f2,f3

    if i == j == 0:
        f = Ac(0,0)
        return (f,)

    if i==0: ## this never happens!
        assert False

    if j==0:
        return SigDel2((i,0),(-i,0))+SigDel2((0,i),(0,-i))

    if i==j:
        return SigDel4((i, i),(-i, i),
                       (i,-i),(-i,-i))

    ## else: (general case)
    fij = SigDel4((i, j),(-i, j),
                  (i,-j),(-i,-j))
    fji = SigDel4((j, i),(-j, i),
                  (j,-i),(-j,-i))
    return fij+fji

def kleinsigfeatures(A,c,i,j):
    nr,nc = A.shape
    def Ac(ii,jj):
        return A[c+ii:nr-c+ii,c+jj:nc-c+jj]

    if i==j:
        f = (Ac(i,i)+Ac(-i,i)+Ac(i,-i)+Ac(-i,-i))/4.
        return (f,)

    fij = ( Ac(i, j)+Ac(-i, j)+Ac(i,-j)+Ac(-i,-j) )/4.
    fji = ( Ac(j, i)+Ac(-j, i)+Ac(j,-i)+Ac(-j,-i) )/4.
    return (fij,fji)

def arrayfeaturesfromannulus(A,ro,ri,fcn_features=annfeatures):
    assert ri <= ro
    flist = []
    for i in range(ri,ro+1):
        for j in range(0,i+1):
            fs = fcn_features(A,ro,i,j)
            flist.extend(fs)
    return np.asarray(flist)

def f_ring(A,ro,ri):
    assert ri <= ro
    flist = []
    for i in range(ri,ro+1):
        fs = ringfeatures(A,ro,i)
        flist.extend(fs)
    return np.asarray(flist)

def f_diamond_ring(A,ro,ri):
    assert ri <= ro
    flist = []
    for i in range(ri,ro+1):
        fs = diamondringfeatures(A,ro,ri,i)
        flist.extend(fs)
    return np.asarray(flist)

def f_zero(A,ro,ri):
    fs = nullfeatures(A,ro)
    return np.asarray((fs,))

def f_ann(A,ro,ri):
    return arrayfeaturesfromannulus(A,ro,ri,fcn_features=annfeatures)

def f_mean(A,ro,ri):
    Z = f_ann(A,ro,ri) ## direct approach might save some memory
    Z = np.mean(Z,0)
    Z = Z.reshape(1,Z.shape[0],Z.shape[1])
    return Z

def f_median(A,ro,ri):
    Z = f_ann(A,ro,ri)
    Z = np.median(Z,0)
    Z = Z.reshape(1,Z.shape[0],Z.shape[1])
    return Z

def f_max(A,ro,ri):
    Z = f_ann(A,ro,ri)
    Z = np.max(Z,0) ## presumably mem-saving opportunities here too
    Z = Z.reshape(1,Z.shape[0],Z.shape[1])
    return Z

def f_dihedral_sigdel(A,ro,ri):
    return arrayfeaturesfromannulus(A,ro,ri,fcn_features=symfeatures)

def f_klein_sigdel(A,ro,ri):
    return arrayfeaturesfromannulus(A,ro,ri,fcn_features=kleinfeatures)

def f_klein_sig(A,ro,ri):
    return arrayfeaturesfromannulus(A,ro,ri,fcn_features=kleinsigfeatures)

def f_dihedral_sig(A,ro,ri):
    return arrayfeaturesfromannulus(A,ro,ri,fcn_features=symsigfeatures)

def f_card(A,ro,ri):
    assert ri <= ro
    fs = diamondringfeatures(A,ro,ri,1)
    return np.asarray(fs)

def unwind(f,ro,ri,f_fcn=f_ann):
    '''map features (f) back to annulus (A)'''
    D = 2*ro+1 # diameter
    A = np.zeros((D,D))
    u = defaultdict(list)
    nfeatures = len(f)

    for i in range(D):
        for j in range (D):
            A = 0*A
            A[i,j]=1
            x = f_fcn(A,ro,ri).flatten()
            if len(x) != nfeatures:
                raise RuntimeError("inconsistent feature length: "
                                   "%d != %d"%(len(x),nfeatures))
            k, = np.nonzero(x)
            if len(k)>1:
                raise RuntimeError("multiple features map to single pixel")
            if len(k)>0:
                u[(i,j)]=k[0]

    kCount = Counter(u.values())

    A = np.zeros((D,D))
    for i,j in u:
        k = u[(i,j)]
        A[i,j] = f[k]/kCount[k]

    return A

def rotarrayfeaturesfromannulus(A,ro,ri):
    ## Return 8 copies of every feature, one for a different
    ## orientation of the image A (one flip, four rot90's)
    assert ri <= ro
    nFeatures = (2*ro+1)**2 - (2*ri-1)**2
    nr = A.shape[0] - 2*ro
    nc = A.shape[1] - 2*ro
    FF = np.zeros((8,nFeatures,nr,nc))

    for k in range(4):
        Arot = np.rot90(A,k=k)
        f = arrayfeaturesfromannulus(Arot,ro,ri)
        for d in range(f.shape[0]):
            fd = np.squeeze(f[d,:,:])
            FF[k,d,:,:] = np.rot90(fd,k=-k)

        Arot = np.fliplr(Arot)
        f = arrayfeaturesfromannulus(Arot,ro,ri)
        for d in range(f.shape[0]):
            fd = np.squeeze(f[d,:,:])
            fd = np.fliplr(fd)
            FF[k+4,d,:,:] = np.rot90(fd,k=-k)

    ### Over-write all those rotations to make everything identical!
    #for k in range(4):
    #    FF[k  ,:,:,:] = FF[0,:,:,:]
    #    FF[k+4,:,:,:] = FF[0,:,:,:]

    F = np.zeros((nFeatures,nr,nc*8))
    for k in range(8):
        F[:,:,nc*k:nc*k+nc] = FF[k,:,:,:]

    return F

AnnulusFunction = {
    ## single feature functions
    "zero":     f_zero,
    "mean":     f_mean,
    "card":     f_card,
    "median":   f_median,
    "meanr":    f_mean,
    "cardr":    f_card,
    "medianr":  f_median,
    ## multiple feature functions
    "rings":    f_ring,
    "diamond":  f_diamond_ring,
    "d4sig":    f_dihedral_sig,
    "k4sig":    f_klein_sig,
    "d4sigdel": f_dihedral_sigdel,
    "k4sigdel": f_klein_sigdel,
    "unsym":    f_ann,
}

def feature_names():
    '''return list of all feature names'''
    return list(AnnulusFunction)

def non_regressive_features():
    '''non-regressive features are meant to be used as predictors directly'''
    ## Note that meanr, medianr, and cardr can be used with a regressor
    return ('mean', 'median','card')

