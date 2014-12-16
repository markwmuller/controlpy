""" Robust control related tools

See the excellent lecture notes by Hannu T. Toivonen "Robust control methods"

(c) 2014 Mark W. Mueller
"""

from __future__ import division, print_function

import numpy as np
import scipy.linalg

import controllability


def controller_H2_state_feedback(A, Bdist, Binput, C1, D12):
    """Solve for the optimal H2 state feedback controller.
    
    A, Bdist, and Binput are system matrices, describing the systems dynamics:
     dx/dt = A*x + Binput*u + Bdist*v
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
    Bdist : (n, m) Matrix
         Input
    Binput : (n, p) Matrix
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

    S = scipy.linalg.solve_continuous_are(A, Binput, C1.T*C1, D12.T*D12)

    K = scipy.linalg.inv(D12.T*D12)*Binput.T*S

    J = np.trace(Bdist.T*S*Bdist)
    
    return K, S, J


# def controller_Hinf_state_feedback(A, Bdist, Binput, C1, D12, stabilityBoundaryEps=1e-16, gammaPrecision=1e-6):
#     """Solve for the optimal H_infinity state feedback controller.
#       
#     A, Bdist, and Binput are system matrices, describing the systems dynamics:
#      dx/dt = A*x + Binput*u + Bdist*v
#      where x is the system state, u is the input, and v is the disturbance
#       
#     The goal is to minimize the output Z, in the H_inf sense, defined as
#      z = C1*x + D12*u
#       
#     Parameters
#     ----------
#     A  : (n, n) Matrix
#          Input
#     Bdist : (n, m) Matrix
#          Input
#     Binput : (n, p) Matrix
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
# #     assert controllability.is_stabilisable(A, Binput), '(A, Binput) must be stabilisable'
# #     assert np.linalg.det(D12.T*D12), 'D12.T*D12 must be invertible'
# #     assert np.max(np.abs(D12.T*C1))==0, 'D12.T*C1 must be zero'
# #     tmp = controllability.unobservable_modes(C1, A, returnEigenValues=True)[1]
# #     if tmp:
# #         assert np.max(np.abs(np.real(tmp)))>0, 'The pair (C1,A) must have no unobservable modes on imag. axis'
#     
#     
#     #First, solve the ARE:
#     # A.T*X+X*A - X*Binput*inv(D12.T*D12)*Binput.T*X + gamma**(-2)*X*Bdist*Bdist.T*X + C1.T*C1 = 0
#     #Let:
#     # R = [[-gamma**(-2)*eye, 0],[0, D12.T*D12]]
#     # B = [Bdist, Binput]
#     # Q = C1.T*C1
#     #then we have to solve
#     # A.T*X+X*A - X*B*inv(R)*B.T*X + Q = 0
#      
#      
#     B = np.matrix(np.zeros([Bdist.shape[0],(Bdist.shape[1]+Binput.shape[1])]))
#     B[:,0:Bdist.shape[1]] = Bdist
#     B[:,Bdist.shape[1]:]  = Binput
#      
#     R = np.matrix(np.zeros([B.shape[1], B.shape[1]]))
#     R[Bdist.shape[1]:,Bdist.shape[1]:] = D12.T*D12
#     Q = C1.T*C1
#     
#     gammaLB = 0
#     gammaUB = 1
#     
#     def has_stable_solution(g, A, B, Q, R, eps):
#         R[:Bdist.shape[1],:Bdist.shape[1]] = -g**(2)*np.eye(Bdist.shape[1], Bdist.shape[1])
#         X = scipy.linalg.solve_continuous_are(A, B, Q, R)
# 
#         CL = A - Binput*np.linalg.inv(D12.T*D12)*Binput.T*X + g**(-2)*Bdist*Bdist.T*X 
#         eigs = np.linalg.eig(CL)[0]
#         
#         return (np.max(np.real(eigs)) < -eps), X
#     
#     #Find an upper bound:
#     counter = 1
#     while has_stable_solution(gammaUB, A, B, Q, R, stabilityBoundaryEps)[0]:
#         gammaUB *= 2
#         counter += 1 
# 
#         assert counter < 1024, 'Exceeded max number of iterations searching for upper gamma bound!'
#         
#     X = 0
#     while (gammaUB-gammaLB)>gammaPrecision:
#         g = 0.5*(gammaUB+gammaLB)
#         stab, X = has_stable_solution(g, A, B, Q, R, stabilityBoundaryEps)
#         if stab:
#             gammaLB = g
#         else:
#             gammaUB = g
#         
#     K = np.linalg.inv(D12.T*D12)*Binput.T*X
#  
#     J = gammaLB
#     return K, X, J
