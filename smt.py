''' Sparse Matrix Transform (SMT)
Approximates a symmetric matrix R (usually positive definite, eg a covariance matrix) 
as a diagonal matrix and a succession of Givens rotations. 
In particular, R = E*D*E', where R is the symmetric matrix, D is a diagonal matrix, and
E = G1*G2*...*GK is an orthogonal matrix that can be expressed 
as a product of K Givens matrices.

Note: here in the comments, I use single quote for transpose, and * for multiply;
that's kind of matlab'ish; translated to python:
by E*D*E', I really mean np.dot(E,np.dot(D,E.T))
'''

import math
import numpy as np

## Convention: To "apply" a matrix to a vector or matrix,
##             is to left multiply by the transpose;
## That is: "A applied to B" is A'*B, or np.dot(A.T,B).
## So, if a class object C has both a matrix() and an apply() method,
## then we have in general that:
## A = C.matrix() and C.apply(B) = np.dot( A.T, B )

def apply_matrix(A,B):
    return np.dot(A.T,B)

## Some generic utilities

def argmax2d(F):
    ''' return indices i,j such that F[i,j] is max of all F values '''
    #actually, this works for 1d or 3d as well...
    return np.unravel_index(np.argmax(F),F.shape)

def rotation_angle(rii,rjj,rij):
    ''' given three matrix elements: rii, rij, rij
    return the rotation angle that transforms the
    off-diagonal element (rij) to zero '''
    theta = 0.5*math.atan2(2*rij,rii-rjj)
    #theta = 0.5*math.atan(2*rij/(rii-rjj))
    return theta

### Wait a minute! Comment doesn't agree with code!!
def criterion(R):
    ''' Return matrix F given by: F[i,j] = R[i,j]**2/(R[i,i]*R[j,j]) for i != j
    (the next matrix element to address will be the one with largest F value)'''
    D = np.diag(R)
    DD = D.reshape(-1,1)*D.reshape(1,-1) ## DD[i,j] = D[i]*D[j] = R[i,i]*R[j,j]
    F = (R-np.diag(D))**2 / DD
    return F

def updatecriterion(F,R,i,j):
    '''
    update criterion matrix F along rows i,j and columns i,j;
    this update is O(d) instead of O(d^2) for recomputing full criterion directly
    '''
    # code is klunky, and on traditional processors, in python, probably no faster
    D = np.diag(R)
    F[:,i] = R[:,i]**2 / (D[i]*D)
    F[:,j] = R[:,j]**2 / (D[j]*D)
    F[i,:] = F[:,i].T
    F[j,:] = F[:,j].T
    F[i,i] = F[j,j] = 0
    return F

## Define classes 

class GivensRotation:
    ''' a Givens rotation is a linear orthogonal matrix operation that operates on 
    only two indices; it is a simple rotation, parameterized by a single angle theta.
    ie, x[i] <-- c_ii*x[i] + c_ij*x[j], and x[j] <-- c_jj*x[j] + c_ji*x[i] 
    where: c_ii = c_jj = cos(theta), and c_ij = -c_ji = sin(theta)
    Directly applying a Givens rotation requires four multiplications.
    sgn=-1 corresponds to rotation by -theta, which is useful as an inverse operation
    '''
    def __init__(self,i,j,theta):
        self.i = i
        self.j = j
        self.c = math.cos(theta)
        self.s = math.sin(theta)

    def ijcs(self):
        '''return tuple of i,j,c,s'''
        return self.i,self.j,self.c,self.s

    def matrix(self,d):
        ''' Implemented mostly for debugging purposes; 
        reutrns the matrix G that corresponds to the the Givens rotation.  
        Specifically, np.dot(G.T,X) is equivalent to Givens.apply(X), 
        except that the matrix multiply would be more computationally expensive.  
        Note also, np.dot(G,X) is equivalent to Givens.apply(X,sgn=-1) 
        '''
        ## if i=0,j=1,d=2; then the 2x2 matrix is of the form [[c,-s],[s,c]].
        i,j,c,s = self.ijcs()
        assert d > max([i,j])
        G = np.identity(d)
        G[i,i] = G[j,j] = c
        G[i,j] = -s
        G[j,i] = s
        return G

    def apply(self,X,sgn=1):
        '''apply rotation to matrix X
        sgn=-1 corresponds to rotation by -theta, 
        which is useful as a transpose or an inverse operation'''
        i,j,c,s = self.ijcs()
        Xi =      c*X[i,:] + sgn*s*X[j,:]
        Xj = -sgn*s*X[i,:] +     c*X[j,:]
        X[i,:] = Xi
        X[j,:] = Xj
        return X

    def multiply(self,X):
        return self.apply(X,sgn=-1)

    def apply_to_vector(self,x,sgn=1):
        '''apply rotaion to vector x'''
        x = self.apply(x.reshape(-1,1),sgn=sgn)
        return x.reshape(-1)

    def multiply_by_vector(self,x):
        return apply_to_vector(self,x,sgn=-1)



def smt_rhat(Glist,D):
    ''' Convert Glist (list of Givens rotations) and diagonal matrix with elements D 
    into a symmetric (covariance) matrix using SMT,
    but without actually making an SMT object.  Useful as a helper function for
    various SMT class functions (inverse covariance, white covariance)
    '''
    ## note D is vector, not matrix; should we assert that?
    R = np.diag(D)
    for G in reversed(Glist):
        ## G*R*G' = G*(G*R')'
        R = G.apply(R.T,sgn=-1)  # G*R'
        R = G.apply(R.T,sgn=-1)  # G*(G*R')'
    return R

## smt_compute is computation without invoking SMT class;
## it does use GivensRotation class, however 
def smt_compute(R,K):
    ''' 
    input covariance matrix R, and number of rotations K;
    output Glist,R with R now a new almost-diagonal matrix
    note: covariance matrix R is over-written, 
    call with R.copy() if you don't want that'''
    Glist = []
    F = criterion(R)
    for k in range(K):
        i,j = argmax2d(F)
        theta = rotation_angle(R[i,i],R[j,j],R[i,j])
        G = GivensRotation(i,j,theta)
        Glist.append( G ) 
        ## update: R <= G'*R*G = G'*(G'*R')'
        R = G.apply(R.T)  # G'*R'
        R = G.apply(R.T)  # G'*(G'*R')' = G'*R*G
        F = updatecriterion(F,R,i,j)
    return Glist,R

class SMT:
    ''' 
    The Sparse Matrix Transform (SMT)
    is a list of K Givens rotations, along with a diagonal matrix D.
    A covariance matrix R can be approximated by EDE', where E = G_1*G_2*...*G_K
    '''
    def __init__(self,d):
        self.d = d
        self.Glist = []
        self.D = np.ones(d)
        
    def copy(self):
        '''return a copy of the SMT object'''
        ## Q: do we need to make copies of G ?
        ## A: don't think so, one never alters a GivensRotation object
        S = SMT(self.d)
        S.Glist = [G for G in self.Glist]
        S.D = self.D.copy()
        return S
    
    @classmethod
    def init_from_covariance(cls,R,K):
        ''' initialize SMT object directly from a covariance matrix: 
        eg, S = SMT.init_from_covariance(R,K)
        input K is number of rotations; recommend something like K=10*d
        note input covariance matrix R is overwritten; 
        call with R.copy() if you want to avoid that
        '''
        d = R.shape[0]
        smt = cls(d)
        Glist,R = smt_compute(R,K)
        smt.Glist = Glist
        smt.D = np.diag(R)
        return smt

    def approx_covariance(self,inverse=False,sqrt=False):
        ''' return the symmetric matrix corresponding to this SMT, 
        it approximates the covariance matrix used to initialize it
        note inverse approx covariance can be computed with virtually the same effort
        and in a very similar way
        '''
        D = 1.0/self.D if inverse else self.D
        if sqrt:
            D = np.sqrt(D)
        R = np.diag(D)
        for G in reversed(self.Glist):
            ## G*R*G' = G*(G*R')'
            R = G.multiply(R.T)  # G*R'
            R = G.multiply(R.T)  # G*(G*R')'
        return R

    def rotations_apply(self,X):
        ''' only apply the rotations to X; over-writing X.
        Equivalent to E'*X, where E given by rotation_matrix() above '''
        for G in self.Glist:
            X = G.apply(X)
        return X

    def rotations_matrix(self):
        ''' Return the orthogonal rotation matrix E = G1*G2*...*GK, 
        with property Rhat = E*D*E'.
        '''
        ## obtained by (E'I).T
        return self.rotations_apply(np.eye(self.d)).T

    def whiten_apply(self,X):
        ''' apply whitening operator to X; over-writing X.
        This is morally (but not strictly) equivalent to multiplying by R^{-1/2}
        Strictly, it is D^{-1/2}*E'*X, where E is the rotation matrix
        '''
        X = self.rotations_apply(X)
        X *= (1.0/np.sqrt(self.D)).reshape(-1,1)
        return X

    def whiten_matrix(self):
        W = self.whiten_apply(np.eye(self.d)).T
        return W

    def mahalanobis(self,X):
        ''' compute Mahalanobis distance (aka, RX)
        Equivalent to x'*R^{-1}*x, for all x in X
        Computed here as magnitude of whitened X'''
        Xshape = X.shape
        X = X.astype(float).reshape(Xshape[0],-1)  ## make sure it's a 2d float array
        X = self.whiten_apply(X)
        r = np.sum(X*X,axis=0)
        r = r.reshape(Xshape[1:])
        return r

    def whiten_matrix_sym(self):
        ''' There are /many/ matrices W that can whiten data; this one is symmetric '''
        return self.approx_covariance(inverse=True,sqrt=True)

    @classmethod
    def testclass(cls):
        '''Run some basic tests to see if class is working'''
        ## Note this routine can also be used for classes
        ## that are derived from this class

        np.random.seed(17)

        d=5
        N=8
        K=12
        Y = np.random.randn(d,N)
        Ro = np.dot(Y,Y.T)/N

        for krot in range(1,K+1):
            obj = cls.init_from_covariance(Ro.copy(),krot)

            Rx = obj.approx_covariance()
            Rxinv = obj.approx_covariance(inverse=True)
            RxinvRo = np.dot(Rxinv,Ro)
            print('%2d Max |RxinvRo - I|= %.6f Max |Ro-Rx|- %.6f' %
                  (krot,
                   np.max(np.abs(RxinvRo-np.eye(d))),
                   np.max(np.abs(Ro-Rx))))

        X = np.random.randn(d,N)

        rmahal = obj.mahalanobis(X)
        print('Mahalanobis:',rmahal[:5])
        print('Mahalanobis:',np.mean(rmahal),'+/-',np.std(rmahal))

        print("Symmetric White Matrix")
        W = obj.whiten_matrix_sym()
        print(W)
        print(W.T.dot(Rx).dot(W).round(decimals=3))
        print(W.T.dot(Ro).dot(W).round(decimals=3))

        print("Asymmetric White Matrix")
        W = obj.whiten_matrix()
        print(W)
        print(W.T.dot(Rx).dot(W).round(decimals=3))
        print(W.T.dot(Ro).dot(W).round(decimals=3))


if __name__ == '__main__':

    SMT.testclass()

    
    


        
        



