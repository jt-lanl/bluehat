'''Variant of SMT that uses fast butterflies

A "fast butterfly" (class Butterfly, below) is a matrix that is
similar to a Givens rotation, in that it only involves two
rows/columns. But where the Givens rotation differs from the identiy
in four positions (ii,ij,ji,jj), the fast Butterfly differs in only
two positions (ij,ji).  Also, whereas the Givens rotation is (as it
says in the name) a rotation, the fast Butterfly does some scaling as
well as rotating. Considering only i,j'th rows/columns, the (now 2x2)
Givens matrix is of the form G=[[c,-s],[s,c]], where c and s correspond
to the cosine and sine of some angle theta.  The fast Butterfly matrix
is of the form B=[[1,a],[b,1]].  

Note that if H is diagonal and G is Givens, then we can write
   BS = HG
where B is fast-Butterfly and S is diagonal; and that B,S are
uniquely defined from this expression

SMT: R = E*D*E', with E = G1*G2*...*GK with D near-diagonal
Kal: R = F*S*D*S*F', with F = B1*B2*...*BK, and S=SK diagonal and D near-diagonal

Thus we have: G1*G2*...*GK = B1*B2*...*BK*SK, which is recursively
satisfied if:
    F1*S1 = G1
    F2*S2 = S1*G2
    ...
    Fk*Sk = S{k-1}*Gk

This tells us how to construct a Kal object from an SMT object, but now
what can we do with it?
1/ Get Rhat = F*S*diag(D)*S*F'
1/ Multiply R by a matrix or vector?
   R*X = F*S*D*S*F'*X
   Z = F'*X = BK'*...*B2'*B1'*X; iteratively apply butterfly
   W = S*D*S*Z; are all diagonals (keep S*D*S around as a single diagonal?)
   R*X = F*W = B1*B2*...*BK*W; backwards iterate and transpose-apply

2/ Get (symmetric) whitening matrix 
   W = F*S*sqrt(D)

2/ Express Rinv as a Kal object?
   We know that Rinv = E*Dinv*E' and E=F*S=B1*B2*...*BK*SK

'''

## NOTES:
## Maybe we want three classes,
## Butterfly, Kal (which is a list of butterflies), and BSMT (butterfly-SMT)
## Similarly for SMT, with an intermedial class for basically Glist

import math
import numpy as np
from .smt import *

class Butterfly(GivensRotation):
    ''' in general, a 'butterfly' is a square matrix B, which is similar to
    a GivensRotation matrix in that it only operates on a pair of rows/columns.
    Looking only at that pair, the 2x2 matrix is of the form [[1,a],[b,1]]. 
    Thus, if y <- Bx is the matrix multiply then:
    y[i] = B_ii*x[i] + B_ij*x[j], and 
    y[j] = B_ji*x[i] + B_jj*x[j], and
    y[k] = x[k] for all k not in {i,j}.
    and x[k] <- x[k] for all k except i,j.
    For our "fast" butterfly, we consider the special case B_ii = B_jj = 1;
    and we write: a=B_ij and b=B_ji.
    This enables the matrix multiplication to proceed with only two scalar multiplications.

    As usual, to "apply" a butterfly to x, we use the transpose: y = B'*x

    
    Inherits methods multiply, apply_to_vector, mutlply_by_vector from GivensRotation.

    An alternative (and truly abstruse) name for this class might be Hesperiidae, 
    a family of butterflies (aka Skippers) known to be particularly quick in flight.

    '''
    def __init__(self,i,j,a,b):
        self.i = i
        self.j = j
        self.a = a
        self.b = b

    def ijab(self):
        return self.i,self.j,self.a,self.b

    def matrix(self,d):
        ## if i=0,j=1,d=2; then the 2x2 matrix is [[1,a],[b,1]]
        i.j,a,b = self.ijab()
        assert d > max([i,j])
        B = np.identity(d)
        B[i,j] = a
        B[j,i] = b
        return B

    def apply(self,X,sgn=1):
        ## equivalent to B'*X
        i,j,a,b = self.ijab()
        if sgn<0:
            ## negative sgn; use transpose, so swap a,b
            a,b = b,a
        Xi = X[i,:] + b*X[j,:]
        Xj = a*X[i,:] + X[j,:]
        X[i,:] = Xi
        X[j,:] = Xj
        return X

    def multiply(self,X):
        return self.apply(X,sgn=-1)


def abuv_from_uvcs(u,v,c,s):
    '''scalar components of BS_from_HG''' 
    a = -(s/c)*u/v
    b =  (s/c)*v/u
    u = u*c
    v = v*c
    return a,b,u,v    

def BS_from_HG(H,G):
    ''' 
    H is vector, corresponding to elements of a diagonal matrix (will be overwritten)
    G is Givens rotation (i,j,c,s)
    B is a fast butterfly transform (i,j,a,b)
    S is vector corresponding to elemenbts of diagonal matrix (over-written H)
    '''
    i,j,c,s = G.ijcs()
    u,v = H[i],H[j]
    a,b,u,v = abuv_from_uvcs(u,v,c,s)
    S=H
    S[i]=u; S[j]=v
    B = Butterfly(i,j,a,b)
    return B,S
    
    
class Kal(SMT):
    ''' Specialized variant of SMT; should give same results, but may be faster[*]

    A flock of butterflies is called a kaleidescope.
    I'm not making this up, but my source is the internet, so I'm not sure.  
    Kal is the name of the class, because kaleidescope is too hard to spell.

    Kal is a collection of K fast butterflies (Blist), 
           plus a vector of multipliers (Smult).  
    Kal can do many of the same things as SMT, 
    but often using only 2*K+d scalar multiplications, vs 4*K+d for SMT.
    Analogy: Kal is to SMT as Butterfly is to GivensRotation.

    ([*] note that it may not actually be much faster on conventional processors; 
    but for many of the operations, fewer scalar multiplications [roughly half]
    are needed; so for onboard processing, using a lower-level language [lower
    than python], it may in fact be faster.)

    Inherits from SMT: whiten_apply, whiten_matrix, mahalanobis from SMT
    '''
    def __init__(self,d):
        self.d = d
        self.Blist = []
        self.Smult = np.ones(d)
        self.D = np.ones(d)

    @classmethod        
    def init_from_SMT(cls,smt):
        kal = cls(smt.d)
        kal.D = smt.D
        for g in smt.Glist:
            B,S = BS_from_HG(kal.Smult,g)
            kal.Blist.append(B)
            kal.Smult = S
        return kal

    @classmethod
    def init_from_covariance(cls,R,K):
        ''' initialize Kal object directly from a covariance matrix: 
        eg, kal = Kal.init_from_covariance(R,K)
        input K is number of rotations; recommend something like K=10*d
        note input covariance matrix R is overwritten; 
        call with R.copy() if you want to avoid that
        '''
        smt = SMT.init_from_covariance(R,K)
        kal = cls.init_from_SMT(smt)
        return kal
    

    def rotations_apply(self,X):
        ''' 
        only apply the rotations to X; over-writing X.
        return [B1*B2*...*BK*S]'*X = S*BK'*...*B2'*B1'*X
        '''
        for B in self.Blist:
            X = B.apply(X)
        X *= self.Smult.reshape(-1,1)
        return X

    def approx_covariance(self,inverse=False,sqrt=False):
        D = 1.0/self.D if inverse else self.D
        D = np.sqrt(D) if sqrt else D
        R = np.diag(self.Smult * D * self.Smult)
        for B in reversed(self.Blist):
            ## B*R*B' = B*(B*R')'
            R = B.multiply(R.T)
            R = B.multiply(R.T)
        return R            

if __name__ == "__main__":

    Kal.testclass()

