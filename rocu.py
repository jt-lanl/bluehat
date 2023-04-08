## ROC Curve Utilities

import numpy as np
import statistics
import math

## Note: three roc() functions are provided, and the first one has a
## keyword so actually I guess there are four.  The first, roc, is the
## fastest and most accurate. Using conservative=True will make it a
## little faster, but not much, and less accurate, but in a
## 'conservative' way.  If tgt and bkg values are all distinct, then
## it should be fully accurate.  The routines roc_OK and roc_FAIR are
## not as fast, but it is more straightforward to follow their
## algorithms. roc_OK is accurate (it should give output essentially
## identical to that of roc -- actually len(fa) will be slightly
## smaller, but still equivalent) but relatively slow; roc_FAIR is
## faster thatn roc_OK (but not as fast as roc) and is fairly
## accurate, not quite as accurate as roc and roc_OK.  Whether it is
## more or less accurate than roc(conservative=True) depends on how
## many bkg points there are, and how often tgt and bkg values exactly
## agree. For roc_FAIR, the len(fa) is smallest, because only one pd
## value for each fa.

def roc(tgt,bkg,conservative=False):
    ''' return fa,pd arrays associated with ROC curve
    this version avoids looping and branching
    assumes tgt,bkg are numpy arrays (needn't be 1D)
    note, conservative=True is faster (and uses less memory), but
    is conservative when it comes to equal-valued tgt,bkg values.
    '''
    NUL_ID = 0
    TGT_ID = 1
    BKG_ID = 2

    tgt = np.array(tgt) ## ensure or convert array to numpy
    bkg = np.array(bkg)

    Nt = tgt.size
    Nb = bkg.size
    id =  np.empty(Nt+Nb+2, dtype=int)   ## id[n] says n'th item is BKG or TGT (or NUL)
    val = np.empty(Nt+Nb+2, dtype=float) ## val[n] is value of n'th item

    ## id/val are long single arrays that concatenate tgt + bkg values
    ## plus 2 extra at the ends, -inf and +inf, to help with bookkeeping
    id[0]=      NUL_ID; val[0] = -np.inf
    id[1:Nt+1]= TGT_ID; val[1:Nt+1] = tgt.flatten()
    id[Nt+1:-1]=BKG_ID; val[Nt+1:-1] = bkg.flatten()
    id[-1]=     NUL_ID; val[-1] = np.inf

    ## sort the values, use argsort so can sort id's as well
    ## stable is important here, for conservative ROC curves
    ndx = np.argsort(val,kind='stable')
    id = id[ndx][:-1]

    fa = np.add.accumulate(id==BKG_ID)  ## fa[n] is how many bkg values below thresh
    pd = np.add.accumulate(id==TGT_ID)

    if not conservative:
        ## Remove values if adjacents are equal, produces diagonals across
        ## stair-steps that are due to equal values of tgt and bkg
        val = val[ndx]
        vneq = val[1:] != val[:-1]  ## checks whether adjacent values are unequal
        fa = fa[vneq]
        pd = pd[vneq]

    fa = 1 - fa/np.max(fa) ## now fa[n] is fraction of bkg values above thresh
    pd = 1 - pd/np.max(pd)

    return fa,pd

def roc_OK(tgt,bkg):
    """
    Given arrays for anomalousness of target and background pixels,
    Compute a false alarm vs detection rate curve.

    Computation here is more direct and straightforward, but uses
    nested while loops with changing conditions, and is not quite as
    performant for large Nb,Nt.
    """

    Nb = bkg.size
    Nt = tgt.size
    bkgs = np.sort(bkg.flatten())
    tgts = np.sort(tgt.flatten())
    fa = [1.0]
    pd = [1.0]
    it=0;
    ib=0;
    indx = 1
    while (it < Nt and ib < Nb):
        th = min([bkgs[ib],tgts[it]]);   ## threshold
        while ( it<Nt and tgts[it] <= th ):
            it += 1
        pd.append(1.0 - it/Nt)  ## pd = #{tgt > th} / Nt
        while( ib<Nb and bkgs[ib] <= th):
            ib += 1
        fa.append(1.0 - ib/Nb)  ## fa = #{bkg > th} / Nb

    pd.append(0)
    fa.append(0)

    fa = np.array(fa,dtype=np.float)
    pd = np.array(pd,dtype=np.float)
    return fa,pd

def roc_FAIR(tgt,bkg):
    """Given arrays for anomalousness of target and background pixels,
    Compute a false alarm vs detection rate curve

    A little faster than roc_OK, but provides a slight upper bound on
    the actual curve, giving only one pd value for each fa value, and that
    is the largest pd associated with that fa.  For a large number of bkg
    values, the AUC from this ROC would be overestimated by 1/(2*Nb), actually
    by a little bit less than this.

    """

    Nb = bkg.size
    Nt = tgt.size
    bkgs = np.sort(bkg.flatten())
    tgts = np.sort(tgt.flatten())
    fa = [1.0]
    pd = [1.0]
    it=0;
    ib=0;
    indx = 1
    while (ib < Nb):
        th = bkgs[ib]; ## threshold
        while( ib<Nb and bkgs[ib] <= th):
            ib += 1
        fa.append(1.0 - ib/Nb)  ## fa = #{bkg > th} / Nb
        while ( it<Nt and tgts[it] <= th ):
            it += 1
        pd.append(1.0 - it/Nt)  ## pd = #{tgt > th} / Nt
        #print((ib,it),end=",")

    fa = np.array(fa,dtype=np.float)
    pd = np.array(pd,dtype=np.float)
    return fa,pd

def roc_OLD(tgt,bkg):
    """Given arrays for anomalousness of target and background pixels,
    Compute a false alarm vs detection rate curve"""

    Nb = bkg.size
    Nt = tgt.size
    bkgs = np.sort(bkg.flatten())
    tgts = np.sort(tgt.flatten())
    fa = np.zeros(Nb)
    pd = np.zeros(Nb)
    it=0;
    ib=0;
    indx=-1;
    while (ib < Nb):
        indx += 1
        th = bkgs[ib]; ## threshold
        #fa[indx] = 1.0 - (ib+0.5)/Nb; ## fa = #{bkg >= th} / nb
        if indx >= Nb:
            print("bkgs: ",bkgs[-5:])
            print("tgts: ",tgts[-5:])
        fa[indx] = 1.0 - (ib+1.0)/(Nb+1.0); ## Laplace-ish; guarantees 0<fa<1, strictly
        while ( it<Nt and tgts[it] <= th ):
            it += 1

        #pd[indx] = 1.0 - (it+1.0)/(Nt+2.0); ## Laplace estimate; guarantees 0 < pd < 1, strictly
        pd[indx] = 1.0 - it/Nt; ## Simple estimate
        while ( ib < Nb and bkgs[ib] <= th ):
            ib += 1

    fa = fa[:indx+1];
    pd = pd[:indx+1];
    return fa,pd

## fapd_fcn_of_fa, fapd_fcn_of_pd (maybe good to call before the fapdsample?)

def fapd_fcn_of_fa(fa,pd):
    '''keep only one fa,pd value per fa; the one with highest pd'''
    ndx = np.empty(shape=fa.shape,dtype=bool)
    ndx[0]=True
    ndx[1:] = fa[1:] != fa[:-1]
    return fa[ndx],pd[ndx]

def fapd_fcn_of_pd(fa,pd):
    '''keep only one fa,pd value per pd; the one with lowest'''
    ndx = np.empty(shape=pd.shape,dtype=bool)
    ndx[-1]=True
    ndx[:-1] = pd[1:] != pd[:-1]
    return fa[ndx],pd[ndx]

# 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
# Returns a positive value, if OAB makes a counter-clockwise turn,
# negative for clockwise turn, and zero if the points are collinear.
def _cross(o, a, b):
    '''internal cross product'''
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def fapd_hull(fa,pd):
    '''
    input ROC curve as arrays fa,pd
    return ROC curve as new arrays fa,pd;
    where the new fa,pd is a subset of the original fa,pd
    corresponding to the convex hull of the points on the curve;
    (and regardless of order of input arrays, output array will
    by in reverse sorted order; ie, first element fa[0]=1,pd[0]=1)
    uses the 'monotone chain' algorithm described in
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
    '''
    assert len(fa)==len(pd)

    hull = []
    for p in reversed(sorted(zip(fa,pd))):
        while len(hull)>1 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    ## more pythonic? kind of abstruse, but it seems to work!
    return tuple(map(np.array,zip(*hull)))

    ## here's what the above code is doing; this is less "elegant"
    ## but it's easier for us humans to read
    #return (np.array([xfa for xfa,_ in hull]),
    #        np.array([xpd for _,xpd in hull]))


def fapdsample(fa,pd,g=1.1):
    ''' return a subsample of fa,pd points; keeps plotter from
    having too much data to work with'''
    if g<=1:
        raise RuntimeError("fapdsample factor less than one: %g"%g)
    n = len(fa)
    ln = 2+int(math.log(n)/math.log(g))
    ndx = n - g ** np.asarray(range(ln))
    ndx = ndx.astype('i')
    ndx[ndx<0]=0
    ndx = np.array(sorted(set(ndx))) ## eliminate duplicates
    fa = fa[ndx]
    pd = pd[ndx]
    return fa,pd

def fa_pdhalf_ok(tgt,bkg):
    '''direct, and a little slower but quite reliable'''
    fa,pd = roc(tgt,bkg)
    return fapdx(fa,pd)

def fa_pdhalf_median(tgt,bkg):
    th = np.median(tgt)
    return np.sum(bkg >= th)/bkg.size

def fa_pdhalf(tgt,bkg,x=0.5):
    ''' return the false alarm rate associated with pd=0.5 '''
    tgt = np.array(tgt)
    bkg = np.array(bkg)
    
    ## If there is an even number of targets, then regular median
    ## may slighlty overestimate min(fa[pd>=0.5]), fix is median_high
    ## which is available via np.percentile()
    thresh = np.percentile(tgt.flatten(),100-100*x,interpolation='higher')

    ## Note if there are a lot of degnerate bkg=tgt values near the
    ## half-point, this may not be truly correct
    fa = np.sum(bkg >= thresh) / bkg.size
    return fa

def fapdx(fa,pd,x=0.5): ### slower than fa_pdhalf (and requires fa,pd)
    ''' return FAR assocated with PD==x '''
    if max(pd) < x:
        return 1.0
    return min( fa[pd >= x] )

def pdfax(fa,pd,x=0.01):
    return max( pd[fa <= x] )

def pd_fax(tgt,bkg,x=0.01):
    thresh = np.percentile(bkg.flatten(),100*(1-x),interpolation='higher')
    #print("bkg thresh:",thresh)
    pd = np.sum(tgt > thresh) / tgt.size
    return pd


def auc_fapd_th(fa,pd,th=1.0):
    ''' return AUC from fa,pd but only over the range 0<=fa<=pd. '''
    assert(fa[0] >= fa[1]) ## assume fa's decrease from 0
    dfa = fa[:-1]-fa[1:]
    spd = pd[:-1]+pd[1:]
    if th<1.0:
        spd[fa[1:]>th]=0
    a = np.sum(dfa*spd)/2
    return a/th

def auc_fapd(fa,pd):
    ''' return AUC from fa,pd (which are assumed to be np.array's) '''
    assert(fa[0] >= fa[1]) ## assume fa's decrease from 0
    dfa = fa[:-1]-fa[1:]
    spd = pd[:-1]+pd[1:]
    a = np.sum(dfa*spd)/2
    return a

def auc(tgt,bkg,th=1.0):
    fa,pd = roc(tgt,bkg)
    return auc_fapd_th(fa,pd,th)
        
def equalize(a):
    '''return an array of same size and shape as a, but with
    values monotonically transformed to a range uniformly 
    distributed between 0 and 1, not including 0 or 1
    '''
    na = a.size
    xarg = np.argsort(a.flatten())
    xrnk = np.zeros(na)
    for i in range(na):
        xrnk[xarg[i]] = (i+0.5)/na
    return xrnk.reshape(a.shape)

