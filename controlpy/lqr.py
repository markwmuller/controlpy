""" Linear quadratic regulator tools.

"""

from __future__ import division, print_function

import numpy as np
import scipy.linalg


def lqr_continuous_time(A,B,Q,R):
    """Solve the continuous time LQR controller for a continuous time system.
    
    A and B are system matrices, describing the systems dynamics:
     dx/dt = A x + B u
    
    The controller minimizes the infinite horizon quadratic cost function:
     cost = integral x.T*Q*x + u.T*R*u
    
    where Q is a positive semidefinite matrix, and R is positive definite matrix.
    
    Returns K, X, eigVals:
    Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
    The optimal input is then computed as:
     input: u = -K*x
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)
    
    #compute the LQR gain
    K = np.dot(scipy.linalg.inv(R),(np.dot(B.T,X)))  # todo! Do this without an explicit inverse...
    
    eigVals, eigVecs = scipy.linalg.eig(A-np.dot(B,K))
    
    return K, X, eigVals



def lqr_discrete_time(A,B,Q,R):
    """Solve the discrete time LQR controller for a discrete time system.
    
    A and B are system matrices, describing the systems dynamics:
     x[k+1] = A x[k] + B u[k]
    
    The controller minimizes the infinite horizon quadratic cost function:
     cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    
    where Q is a positive semidefinite matrix, and R is positive definite matrix.
    
    Returns K, X, eigVals:
    Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
    The optimal input is then computed as:
     input: u = -K*x
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)
    
    #compute the LQR gain
    K = np.dot(scipy.linalg.inv(np.dot(np.dot(B.T,X),B)+R),(np.dot(np.dot(B.T,X),A)))  # todo! Remove inverse.
    
    eigVals, eigVecs = scipy.linalg.eig(A-np.dot(B,K))
    
    return K, X, eigVals


