from __future__ import division, print_function

import numpy as np
import scipy.linalg



def compute_lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
    
    dx/dt = A x + B u
    
    cost = integral x.T*Q*x + u.T*R*u
    
    Returns gain K, X, and the closed loop system eigenvalues
    input: u = -K*x
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, X, eigVals



def compute_lqr_discrete_time(A,B,Q,R):
    """Solve the discrete time lqr controller.
    
    
    x[k+1] = A x[k] + B u[k]
    
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, X, eigVals


def steady_state_kalman_filter(A, H, Q, R):
    X = np.matrix(scipy.linalg.solve_discrete_are(A.T, H.T, Q, R))
    
    #compute the kalman filter gain
    K = np.matrix(X*H.T*scipy.linalg.inv(H*X*H.T + R))
    
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, X, eigVals
    

    
def gramian_controllability(A, B):
    '''Compute the controllability gramian of the stable continuous time system.
    
    dx = A*x + B*u
    
    '''
    eigVals, eigVecs = scipy.linalg.eig(A)
    if np.max(np.real(eigVals)) >= 0:
        print('Cannot compute gramian for A, has an eigen value with real part:',np.max(np.real(eigVals)))
        return None
    
    Wc = scipy.linalg.solve_lyapunov(A, -B*B.T)
    return Wc
    
    
    
    
    
    
    
    
    
