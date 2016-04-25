'''A simple example script, which implements an LQR controller for a double integrator.
'''

from __future__ import print_function, division

import controlpy
import cvxpy

import numpy as np

#compute full state h-inf gain using LMIs:
def get_hinf(A, Binput, Bdist, C1, D12, gammaLB=0, gammaUB=np.inf, gammaRelTol=1e-3):
    K = None
    
    n = A.shape[0]  #num states
    m = Binput.shape[1]  #num control inputs
    q = C1.shape[0]  #num outputs to "be kept small"

    def has_solution(g):
        Q = cvxpy.Semidef(n)
        Y = cvxpy.Variable(m,n)
        
        m1 = A*Q + Q*A.T + Binput*Y + Y.T*Binput.T + Bdist*Bdist.T 
        m2 = (C1*Q+D12*Y).T
        m3 = (C1*Q+D12*Y)
        m4 = - g**2*np.identity(q)
        
        tmp1 = cvxpy.hstack(m1, m2)
        tmp2 = cvxpy.hstack(m3, m4)
        tmp  = cvxpy.vstack(tmp1, tmp2)

        constraints = [tmp == -cvxpy.Semidef(n+q)]

        obj = cvxpy.Minimize(1)

        prob = cvxpy.Problem(obj, constraints)
        
        prob.solve()

        if not prob.value == 1:
            #infeasible
            return False, None
        else:
            return True, -Y.value*np.linalg.inv(Q.value)
        
        
    if np.isinf(gammaUB):
        #automatically choose an UB
        gammaUB = np.max([1, gammaLB])
        counter = 1
        while True:
            ok, K = has_solution(gammaUB)
            if ok:
                break
            gammaUB *= 2

            assert counter < 1024, 'Exceeded max number of iterations searching for upper gamma bound!'
        
    while gammaUB - gammaLB > gammaRelTol*gammaUB:
        gammaTry = 0.5*(gammaUB + gammaLB)
        
        hasSol, Ktry = has_solution(gammaTry)
        if hasSol:
            gammaUB = gammaTry
            K = Ktry
        else:
            gammaLB = gammaTry


    return K, gammaUB


# Example system is a double integrator:
A = np.matrix([[-10,1],[0,-1]],dtype=np.float)
B = np.matrix([[0],[1]],dtype=np.float)

Bdist = np.matrix([[1,0],[0,1]],dtype=np.float)

C = np.matrix([[1,0],[0,1],[0,0]],dtype=np.float)
D12 = np.matrix([[0,0,1]],dtype=np.float).T

gmin = 0
gmax = np.inf

# Compute the LQR controller
k_H2, _, _ = controlpy.synthesis.controller_H2_state_feedback(A, B, Bdist, C, D12, useLMI=True)
k_Hinf, _, J = controlpy.synthesis.controller_Hinf_state_feedback(A, B, Bdist, C, D12, gammaLB=gmin, gammaUB=gmax)

print('H2:')
print(k_H2)
print('Hinf:', J)
print(k_Hinf)

k_Hnew, g = get_hinf(A, B, Bdist, C, D12, gammaLB=gmin, gammaUB=gmax)

print('new lmi:', g)
print(k_Hnew)

print('H2:      ', controlpy.analysis.system_norm_Hinf(A-B*k_H2, Bdist, C-D12.dot(k_H2)), '\t', controlpy.analysis.system_norm_Hinf_LMI(A-B*k_H2, Bdist, C - D12.dot(k_H2)))
print('ricatti: ', controlpy.analysis.system_norm_Hinf(A-B*k_Hinf, Bdist, C-D12.dot(k_Hinf)), '\t', controlpy.analysis.system_norm_Hinf_LMI(A-B*k_Hinf, Bdist, C - D12.dot(k_Hinf)))
print('new:     ', controlpy.analysis.system_norm_Hinf(A-B*k_Hnew, Bdist, C-D12.dot(k_Hnew)), '\t', controlpy.analysis.system_norm_Hinf_LMI(A-B*k_Hnew, Bdist, C - D12.dot(k_Hnew)))



