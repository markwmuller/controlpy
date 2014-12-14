""" Robust control related tools

See the excellent lecture notes by Hannu T. Toivonen "Robust control methods"

"""

from __future__ import division, print_function

import numpy as np
import scipy.linalg


def controller_H2_state_feedback(A, B1, B2, C1, D12):
    """Solve for the optimal H2 state feedback controller.
    
    A, B1, and B2 are system matrices, describing the systems dynamics:
     dx/dt = A*x + B2*v + B1*u
     where x is the system state, u is the input, and v is the disturbance
    
    The goal is to minimize the output Z, defined as
     z = C1*x + D12*u
     
    The optimal output is given by:
     u = - K*x
    
    This is related to the LQR problem, where the state cost matrix is Q and
    the input cost matrix is R, then:
     C1 = [[sqrt(Q)], [0]] and D = [[0], [sqrt(D12)]]
    With sqrt(Q).T*sqrt(Q) = Q
    
    Parameters
    ----------
    A  : (n, n) Matrix
         Input
    B1 : (n, m) Matrix
         Input
    B2 : (n, p) Matrix
         Input
    C1 : (n, q) Matrix
         Input
    D12: (q, p) Matrix
         Input

    Returns
    -------
    K : (m, n) Matrix
        H2 optimal controller gain
    S : (n, n) Matrix
        Solution to the Ricatti equation
    J : Minimum cost value
    
    """

    S = scipy.linalg.solve_continuous_are(A, B2, C1.T*C1, D12.T*D12)

    K = scipy.linalg.inv(D12.T*D12)*B2.T*S

    J = np.trace(B1.T*S*B1)
    
    return K, S, J


# def controller_Hinf_state_feedback(A, B1, B2, C1, D12, gamma):
#     """Solve for the optimal H_infinity state feedback controller.
#      
#     A, B1, and B2 are system matrices, describing the systems dynamics:
#      dx/dt = A*x + B2*v + B1*u
#      where x is the system state, u is the input, and v is the disturbance
#      
#     The goal is to minimize the output Z, in the H_inf sense, defined as
#      z = C1*x + D12*u
#      
#     Parameters
#     ----------
#     A  : (n, n) Matrix
#          Input
#     B1 : (n, m) Matrix
#          Input
#     B2 : (n, p) Matrix
#          Input
#     C1 : (n, q) Matrix
#          Input
#     D12: (q, p) Matrix
#          Input
#  
#     Returns
#     -------
#     K : (m, n) Matrix
#         H2 optimal controller gain
#     S : (n, n) Matrix
#         Solution to the Ricatti equation
#     J : Minimum cost value
#      
#     """
#     
#     #First, solve the ARE:
#     # A.T*X+X*A - X*B2*inv(D12.T*D12)*B2.T*X + gamma**(-2)*X*B1*B1.T*X + C1.T*C1 = 0
#     #Let:
#     # R = [[-gamma**(-2)*eye, 0],[0, D12.T*D12]]
#     # B = [B1, B2]
#     # Q = C1.T*C1
#     #then we have to solve
#     # A.T*X+X*A - X*B*inv(R)*B.T*X + Q = 0
#     
#     
#     B = np.matrix(np.zeros([B1.shape[0],(B1.shape[1]+B2.shape[1])]))
#     B[:,0:B1.shape[1]] = B1
#     B[:,B1.shape[1]:]  = B2
#     
#     R = np.matrix(np.zeros([B.shape[1], B.shape[1]]))
#     R[:B1.shape[1],:B1.shape[1]] = -gamma**(2)*np.eye(B1.shape[1], B1.shape[1])
#     R[B1.shape[1]:,B1.shape[1]:] = D12.T*D12
#     
#     X = scipy.linalg.solve_continuous_are(A, B, C1.T*C1, R)
#     
#     CL = A - B2*np.linalg.inv(D12.T*D12)*B2.T*X + gamma**(-2)*B1*B1.T*X
#     eigs = np.linalg.eig(CL)[0]
# 
#     print(np.max(np.real(eigs))) # This should be strictly negative
#     
#     K = np.linalg.inv(D12.T*D12)*B2.T*X
# 
#     J = gamma
#     return K, X, J
