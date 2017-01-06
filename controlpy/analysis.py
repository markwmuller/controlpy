""" Tools for analysing LTI systems.

(c) 2014 Mark W. Mueller
"""

import numpy as np
import scipy.linalg
import scipy.integrate


def is_hurwitz(A, tolerance = 1e-9):
    '''Test whether the matrix A is Hurwitz (i.e. asymptotically stable).
    
    tolerance defines the minimum distance we should be from the imaginary axis 
     to be considered stable.
    
    '''
    return max(np.real(np.linalg.eig(A)[0])) < -np.abs(tolerance)


def uncontrollable_modes(A, B, returnEigenValues = False, tolerance=1e-9):
    '''Returns all the uncontrollable modes of the pair A,B.
    
    tolerance defines the minimum distance we should be from the imaginary axis 
     to be considered stable.
    
    Does the PBH test for controllability for the system:
     dx = A*x + B*u
    
    Returns a list of the uncontrollable modes, and (optionally) 
    the corresponding eigenvalues.
    
    See Callier & Desoer "Linear System Theory", P. 253
    
    NOTE!: This can't work if we have repeated eigen-values! TODO FIXME!
    '''

    assert A.shape[0]==A.shape[1], "Matrix A is not square"
    assert A.shape[0]==B.shape[0], "Matrices A and B do not align"

    nStates = A.shape[0]
    nInputs = B.shape[1]

    eVal, eVec = np.linalg.eig(np.matrix(A)) # todo, matrix cast is ugly.

    uncontrollableModes = []
    uncontrollableEigenValues = []

    for e,v in zip(eVal, eVec.T):
        M = np.matrix(np.zeros([nStates,(nStates+nInputs)]), dtype=complex)
        M[:,:nStates] = e*np.identity(nStates) - A
        M[:,nStates:] = B
        
        s = np.linalg.svd(M, compute_uv=False)
        if min(s) <= tolerance: 
            uncontrollableModes.append(v.T[:,0])
            uncontrollableEigenValues.append(e)

    if returnEigenValues:
        return uncontrollableModes, uncontrollableEigenValues
    else:
        return uncontrollableModes
    


def is_controllable(A, B, tolerance=1e-9):
    '''Compute whether the pair (A,B) is controllable.
    tolerance defines the minimum distance we should be from the imaginary axis 
     to be considered stable.
    
    Returns True if controllable, False otherwise.
    '''

    if uncontrollable_modes(A, B, tolerance=tolerance):
        return False
    else:
        return True



def is_stabilisable(A, B):
    '''Compute whether the pair (A,B) is stabilisable.

    Returns True if stabilisable, False otherwise.
    '''

    modes, eigVals = uncontrollable_modes(A, B, returnEigenValues=True)
    if not modes: 
        return True  #controllable => stabilisable
    
    if max(np.real(eigVals)) >= 0:
        return False
    else:
        return True


def controllability_gramian(A, B, T = np.inf):
    '''Compute the causal controllability Gramian of the continuous time system.
    
    The system is described as
     dx = A*x + B*u
     
    T is the horizon over which to compute the Gramian. If not specified, the 
    infinite horizon Gramian is computed. Note that the infinite horizon Gramian
    only exists for asymptotically stable systems.
    
    If T is specified, we compute the Gramian as
     Wc = integrate exp(A*t)*B*B.H*exp(A.H*t) dt 
    
    Returns the matrix Wc.
    '''
    
    assert A.shape[0]==A.shape[1], "Matrix A is not square"
    assert A.shape[0]==B.shape[0], "Matrix A and B do not align"

    if not np.isfinite(T):
        #Infinite time Gramian:
        assert is_hurwitz(A), "Can only compute infinite horizon Gramian for a stable system."
        
        Wc = scipy.linalg.solve_lyapunov(A, -B*B.T)
        return Wc
    
    # We need to solve the finite time Gramian
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

    return uncontrollable_modes(A.conj().T, C.conj().T, returnEigenValues)


def is_observable(C, A):
    '''Compute whether the pair (C,A) is observable.
    
    Returns True if observable, False otherwise.
    '''
    
    return is_controllable(A.conj().T, C.conj().T)


def is_detectable(C, A):
    '''Compute whether the pair (C,A) is detectable.

    Returns True if detectable, False otherwise.
    '''

    return is_stabilisable(A.conj().T, C.conj().T)


#TODO
# def observability_gramian(A, B, T = np.inf):
#     '''Compute the observability Gramian of the continuous time system.
#     
#     The system is described as
#      dx = A*x + B*u
#      
#     T is the horizon over which to compute the Gramian. If not specified, the 
#     infinite horizon Gramian is computed. Note that the infinite horizon Gramian
#     only exists for asymptotically stable systems.
#     
#     If T is specified, we compute the Gramian as
#      Wc = integrate exp(A*t)*B*B.H*exp(A.H*t) dt 
#     
#     Returns the matrix Wc.
#     '''
#     
#     assert A.shape[0]==A.shape[1], "Matrix A is not square"
#     assert A.shape[0]==B.shape[0], "Matrix A and B do not align"
# 
#     if not np.isfinite(T):
#         #Infinite time Gramian:
#         eigVals, eigVecs = scipy.linalg.eig(A)
#         assert np.max(np.real(eigVals)) < 0, "Can only compute infinite horizon Gramian for a stable system."
#         
#         Wc = scipy.linalg.solve_lyapunov(A, -B*B.T)
#         return Wc
#     
#     # We need to solve the finite time Gramian
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


def system_norm_H2(Acl, Bdisturbance, C):
    '''Compute a system's H2 norm.
    
    Acl, Bdisturbance are system matrices, describing the systems dynamics:
     dx/dt = Acl*x  + Bdisturbance*v
    where x is the system state and v is the disturbance.
    
    The system output is:
     z = C*x
    
    The matrix Acl must be Hurwitz for the H2 norm to be finite. 
     
    Parameters
    ----------
    A  : (n, n) Matrix, 
         Input
    Bdisturbance : (n, m) Matrix
         Input
    C : (n, q) Matrix
         Input

    Returns
    -------
    J2 : Systems H2 norm.
    '''
    
    if not is_hurwitz(Acl):
        return np.inf
    
    #first, compute the controllability Gramian of (Acl, Bdisturbance)
    P = controllability_gramian(Acl, Bdisturbance)
    
    #output the gain
    return np.sqrt(np.trace(C*P*C.T))
    

def system_norm_Hinf(Acl, Bdisturbance, C, D = None, lowerBound = 0, upperBound = np.inf, relTolerance = 1e-3):
    '''Compute a system's Hinfinity norm.
    
    Acl, Bdisturbance are system matrices, describing the systems dynamics:
     dx/dt = Acl*x  + Bdisturbance*v
    where x is the system state and v is the disturbance.
    
    The system output is:
     z = C*x + D*v
    
    The matrix Acl must be Hurwitz for the Hinf norm to be finite. 
    
    The norm is found by iterating over the Riccati equation. The search can 
    be sped up by providing lower and upper bounds for the norm. If ommitted, 
    these are determined automatically. 
    The search proceeds via bisection, and terminates when a specified relative
    tolerance is achieved.
     
    Parameters
    ----------
    A  : (n, n) Matrix
         Input
    Bdisturbance : (n, m) Matrix
         Input
    C : (q, n) Matrix
         Input
    D : (q,m) Matrix
         Input (optional)
    lowerBound: float
         Input (optional)
    upperBound: float 
         Input (optional)
    relTolerance: float
         Input (optional)

    Returns
    -------
    Jinf : Systems Hinf norm.
    
    '''

    if not is_hurwitz(Acl):
        return np.inf
    
    eps = 1e-10
    
    if D is None:
        #construct a fake feed-through matrix
        D = np.matrix(np.zeros([C.shape[0], Bdisturbance.shape[1]]))
    

    def test_upper_bound(gamma, A, B, C, D):
        '''Is the given gamma an upper bound for the Hinf gain?
        '''
        #Construct the R matrix:
        Rric = -gamma**2*np.matrix(np.eye(D.shape[1],D.shape[1])) + D.T*D
        #test that Rric is negative definite
        eigsR = np.linalg.eig(Rric)[0]
        if max(np.real(eigsR)) > -eps:
            return False, None
        
        #matrices for the Ricatti equation:
        Aric = A - B*np.linalg.inv(Rric)*D.T*C
        Bric = B
        Qric = C.T*C - C.T*D*np.linalg.inv(Rric)*D.T*C

        try:
            X = scipy.linalg.solve_continuous_are(Aric, Bric, Qric, Rric)
        except np.linalg.linalg.LinAlgError:
            #Couldn't solve
            return False, None
                 
        eigsX = np.linalg.eig(X)[0]
        if (np.min(np.real(eigsX)) < 0) or (np.sum(np.abs(np.imag(eigsX)))>eps):
            #The ARE has to return a pos. semidefinite solution, but X is not
            return False, None  
  
        CL = A + B*np.linalg.inv(-Rric)*(B.T.dot(X) + D.T.dot(C))
        eigs = np.linalg.eig(CL)[0]
          
        return (np.max(np.real(eigs)) < -eps), X
    
    #our ouptut ricatti solution
    X = None
    
    #Are we supplied an upper bound? 
    if not np.isfinite(upperBound):
        upperBound = max([1,lowerBound])
        counter = 1
        while True:
            isOK, X2 = test_upper_bound(upperBound, Acl, Bdisturbance, C, D)

            if isOK:
                X = X2.copy()
                break

            upperBound *= 2
            counter += 1
            assert counter<1024, 'Exceeded max. number of iterations searching for upper bound'
            
    #perform a bisection search to find the gain:
    while (upperBound-lowerBound)>relTolerance*upperBound:
        g = 0.5*(upperBound+lowerBound)
         
        stab, X2 = test_upper_bound(g, Acl, Bdisturbance, C, D)
        if stab:
            upperBound = g
            X = X2
        else:
            lowerBound = g
     
    assert X is not None, 'No solution found! Check supplied upper bound'
    
    return upperBound
    


def discretise_time(A, B, dt):
    '''Compute the exact discretization of the continuous system A,B.
    
    Goes from a description 
     d/dt x(t) = A*x(t) + B*u(t)
     u(t)  = ud[k] for t in [k*dt, (k+1)*dt)
    to the description
     xd[k+1] = Ad*xd[k] + Bd*ud[k]
    where
     xd[k] := x(k*dt)
     
    Returns: Ad, Bd
    '''
    
    nstates = A.shape[0]
    ninputs = B.shape[1]

    M = np.matrix(np.zeros([nstates+ninputs,nstates+ninputs]))
    M[:nstates,:nstates] = A
    M[:nstates, nstates:] = B
    
    Md = scipy.linalg.expm(M*dt)
    Ad = Md[:nstates, :nstates]
    Bd = Md[:nstates, nstates:]

    return Ad, Bd
    


    
    
