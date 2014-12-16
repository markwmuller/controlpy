""" Tests related to the controllability of an LTI system

(c) 2014 Mark W. Mueller
"""

from __future__ import division

import numpy as np
import scipy.linalg
import scipy.integrate


def uncontrollable_modes(A, B, returnEigenValues = False):
    '''Returns all the uncontrollable modes of the pair A,B.
    
    Does the PBH test for controllability for the system:
     dx = A*x + B*u
    
    Returns a list of the uncontrollable modes, and (optionally) 
    the corresponding eigenvalues.
    
    See Callier & Desoer "Linear System Theory", P. 253
    '''

    assert A.shape[0]==A.shape[1], "Matrix A is not square"
    assert A.shape[0]==B.shape[0], "Matrices A and B do not align"

    nStates = A.shape[0]
    nInputs = B.shape[1]

    eVal, eVec = np.linalg.eig(A)

    uncontrollableModes = []
    uncontrollableEigenValues = []

    for e,v in zip(eVal, eVec.T):
        M = np.matrix(np.zeros([nStates,(nStates+nInputs)]), dtype=complex)
        M[:,:nStates] = e*np.eye(nStates,nStates) - A
        M[:,nStates:] = B
        
        s = np.linalg.svd(M, compute_uv=False)
        if min(s) == 0: 
            uncontrollableModes.append(v.T[:,0])
            uncontrollableEigenValues.append(e)

    if returnEigenValues:
        return uncontrollableModes, uncontrollableEigenValues
    else:
        return uncontrollableModes
    


def is_controllable(A, B):
    '''Compute whether the pair (A,B) is controllable.
    
    Returns True if controllable, False otherwise.
    '''

    if uncontrollable_modes(A, B):
        return False
    else:
        return True



def is_stabilisable(A, B):
    '''Compute whether the pair (A,B) is stabilisable.
    '''

    modes, eigVals = uncontrollable_modes(A, B, returnEigenValues=True)
    if not modes: 
        return True  #controllable => stabilisable
    
    if max(np.real(eigVals)) >= 0:
        return False
    else:
        return True


def controllability_gramian(A, B, T = np.inf):
    '''Compute the controllability gramian of the continuous time system.
    
    The system is described as
     dx = A*x + B*u
     
    T is the horizon over which to compute the gramian. If not specified, the 
    infinite horizon gramian is computed. Note that the infinite horizon grammian
    only exists for asymptotically stable systems.
    
    If T is specified, we compute the gramian as
     Wc = integrate exp(A*t)*B*B.H*exp(A.H*t) dt 
    
    Returns the matrix Wc.
    '''
    
    assert A.shape[0]==A.shape[1], "Matrix A is not square"
    assert A.shape[0]==B.shape[0], "Matrix A and B do not align"

    if not np.isfinite(T):
        #Infinite time gramian:
        eigVals, eigVecs = scipy.linalg.eig(A)
        assert np.max(np.real(eigVals)) < 0, "Can only compute infinite horizon gramian for a stable system."
        
        Wc = scipy.linalg.solve_lyapunov(A, -B*B.T)
        return Wc
    
    # We need to solve the finite time gramian
    # Boils down to solving an ODE:
    A = np.array(A,dtype=float)
    B = np.array(B,dtype=float)
    T = np.float(T)
    
    def gramian_ode(y, t0, A, B):
        temp = np.dot(scipy.linalg.expm(A*t0),B)
        dQ = np.dot(temp,np.conj(temp.T))
         
        return dQ.reshape((A.shape[0]**2,1))[:,0]
     
    y0 = np.zeros([A.shape[0]**2,1])[:,0]
    out = scipy.integrate.odeint(gramian_ode, y0, [0,T], args=(A,B))
    Q = out[1,:].reshape([A.shape[0], A.shape[0]])
    return Q


def unobservable_modes(C, A, returnEigenValues = False):
    '''Returns all the unobservable modes of the pair A,C.
    
    Does the PBH test for observability for the system:
     dx = A*x
     y  = C*x
    
    Returns a list of the unobservable modes, and (optionally) 
    the corresponding eigenvalues.
    
    See Callier & Desoer "Linear System Theory", P. 253
    '''

    return uncontrollable_modes(A.getH(), C.getH(), returnEigenValues)


def is_observable(C, A):
    '''Compute whether the pair (C,A) is observable.
    
    Returns True if observable, False otherwise.
    '''
    
    return is_controllable(A.getH(), C.getH())


def is_detectable(C, A):
    '''Compute whether the pair (C,A) is detectable.
    '''

    return is_stabilisable(A.getH(), C.getH())


#TODO
# def observability_gramian(A, B, T = np.inf):
#     '''Compute the observability gramian of the continuous time system.
#     
#     The system is described as
#      dx = A*x + B*u
#      
#     T is the horizon over which to compute the gramian. If not specified, the 
#     infinite horizon gramian is computed. Note that the infinite horizon grammian
#     only exists for asymptotically stable systems.
#     
#     If T is specified, we compute the gramian as
#      Wc = integrate exp(A*t)*B*B.H*exp(A.H*t) dt 
#     
#     Returns the matrix Wc.
#     '''
#     
#     assert A.shape[0]==A.shape[1], "Matrix A is not square"
#     assert A.shape[0]==B.shape[0], "Matrix A and B do not align"
# 
#     if not np.isfinite(T):
#         #Infinite time gramian:
#         eigVals, eigVecs = scipy.linalg.eig(A)
#         assert np.max(np.real(eigVals)) < 0, "Can only compute infinite horizon gramian for a stable system."
#         
#         Wc = scipy.linalg.solve_lyapunov(A, -B*B.T)
#         return Wc
#     
#     # We need to solve the finite time gramian
#     # Boils down to solving an ODE:
#     A = np.array(A,dtype=float)
#     B = np.array(B,dtype=float)
#     T = np.float(T)
#     
#     def gramian_ode(y, t0, A, B):
#         temp = np.dot(scipy.linalg.expm(A*t0),B)
#         dQ = np.dot(temp,np.conj(temp.T))
#          
#         return dQ.reshape((A.shape[0]**2,1))[:,0]
#      
#     y0 = np.zeros([A.shape[0]**2,1])[:,0]
#     out = scipy.integrate.odeint(gramian_ode, y0, [0,T], args=(A,B))
#     Q = out[1,:].reshape([A.shape[0], A.shape[0]])
#     return Q