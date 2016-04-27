from __future__ import print_function, division

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

np.random.seed(1234)

import controlpy

import unittest

import cvxpy
def synth_h2_state_feedback_LMI(A, Binput, Bdist, C1, D12):
    #Dullerud p 217 (?)
    
    n = A.shape[0]  #num states
    m = Binput.shape[1]  #num control inputs
    q = C1.shape[0]  #num outputs to "be kept small"

    X = cvxpy.Variable(n,n)
    Y = cvxpy.Variable(m,n)
    Z = cvxpy.Variable(q,q)
    
    tmp1 = cvxpy.hstack(X, (C1*X+D12*Y).T)
    tmp2 = cvxpy.hstack((C1*X+D12*Y), Z)
    tmp  = cvxpy.vstack(tmp1, tmp2)

    constraints = [A*X + Binput*Y + X*A.T + Y.T*Binput.T + Bdist*Bdist.T == -cvxpy.Semidef(n),
                   tmp == cvxpy.Semidef(n+q),
                  ]

    obj = cvxpy.Minimize(cvxpy.trace(Z))

    prob = cvxpy.Problem(obj, constraints)
    
    prob.solve(solver='CVXOPT', kktsolver='robust')
    
    K = -Y.value*np.linalg.inv(X.value)
    return K


class TestSynthesis(unittest.TestCase):

    def test_h2(self):
        for matType in [np.matrix, np.array]:
            A = matType([[1,2],[0,3]])
            B = matType([[0,1]]).T
            Bdist = matType([[0,1]]).T
            C1 = matType([[1,0],[0,1],[0,0]])
            D12= matType([[0,0,1]]).T
            
            KLMI = synth_h2_state_feedback_LMI(A, B, Bdist, C1, D12)
            K2, X2, J2 = controlpy.synthesis.controller_H2_state_feedback(A, B, Bdist, C1, D12)
            
            self.assertLess(np.linalg.norm(K2-KLMI), 1e-3*np.linalg.norm(K2)) #note sign difference
            



if __name__ == '__main__':
    np.random.seed(1)
    unittest.main()
