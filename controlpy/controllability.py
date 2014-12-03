""" Linear quadratic regulator tools.

"""

from __future__ import division, print_function

import numpy as np
import scipy.linalg
import scipy.integrate

def uncontrollable_modes(A, B):
    '''Returns all the uncontrollable modes of the pair A,B.
    
    Does the PBH test for controllability for the system:
     dx = A*x + B*u
    
    returns a list of the uncontrollable modes.
    '''

    assert A.shape[0]==A.shape[1], "Matrix A is not square"
    assert A.shape[0]==B.shape[0], "Matrix A and B do not align"

    nStates = A.shape[0]
    nInputs = B.shape[1]

    eVal, eVec = np.linalg.eig(A)
    print(eVec[:,0])

    uncontrollableModes = []

    for e,v in zip(eVal, eVec.T):
        M = np.matrix(np.zeros([nStates,(nStates+nInputs)]), dtype=complex)
        M[:,:nStates] = e*np.eye(nStates,nStates) - A
        M[:,nStates:] = B
        
        s = np.linalg.svd(M, compute_uv=False)
        if min(s) == 0: 
            uncontrollableModes.append(v.T[:,0])

    return uncontrollableModes


def controllability_gramian(A, B, T = None):
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

    if T is None:
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
