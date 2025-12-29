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

    X = cvxpy.Variable((n,n))
    Y = cvxpy.Variable((m,n))
    Z = cvxpy.Variable((q,q))
    
    tmp1 = cvxpy.hstack([X, (C1@X+D12@Y).T])
    tmp2 = cvxpy.hstack([(C1@X+D12@Y), Z])
    tmp  = cvxpy.vstack([tmp1, tmp2])

    constraints = [A@X + Binput@Y + X@A.T + Y.T@Binput.T + Bdist@Bdist.T == -cvxpy.Variable((n,n), PSD=True),
                   tmp == cvxpy.Variable((n+q,n+q), PSD=True),
                  ]

    obj = cvxpy.Minimize(cvxpy.trace(Z))

    prob = cvxpy.Problem(obj, constraints)
    
    prob.solve()
    
    K = -Y.value@np.linalg.inv(X.value)
    return K


class TestSynthesis(unittest.TestCase):

    def test_h2(self):
        ''' Tests whether we get same H2 performance when we 
            synthesize using Riccatti method, as when we do LMI
        '''
        for matType in [np.matrix, np.array]:
            A = matType([[1,2],[0,3]])
            B = matType([[0,1]]).T
            Bdist = matType([[0,1]]).T
            C1 = matType([[1,0],[0,1],[0,0]])
            D12= matType([[0,0,1]]).T
            
            KLMI = synth_h2_state_feedback_LMI(A, B, Bdist, C1, D12)
            K2, X2, J2 = controlpy.synthesis.controller_H2_state_feedback(A, B, Bdist, C1, D12)
            
            self.assertLess(np.linalg.norm(K2-KLMI), 1e-3*np.linalg.norm(K2))
            
        for nStates in [1,2,4,8]:
            for nInput in [1,3,5]:
                A = np.random.normal(size=[nStates,nStates])
                B = np.random.normal(size=[nStates,nInput])
                Bdist = np.random.normal(size=[nStates,nInput])
                nOut = nInput + nStates
                C1 = np.random.normal(size=[nOut, nStates])
                C1[nStates:,:] = 0
                D12= np.random.normal(size=[nOut, nInput])
                D12[:nStates,:] = 0
                
                KLMI = synth_h2_state_feedback_LMI(A, B, Bdist, C1, D12)
                K2, X2, J2 = controlpy.synthesis.controller_H2_state_feedback(A, B, Bdist, C1, D12)
                
                # print(KLMI)
                # print(K2)
                
                c1 = controlpy.analysis.system_norm_H2(A-B@KLMI, Bdist, C1)
                c2 = controlpy.analysis.system_norm_H2(A-B@K2, Bdist, C1)
                
                print(c1,c2)

                closeToInfty = 100
                if c1 > closeToInfty:
                    self.assertGreater(c2, closeToInfty)
                elif c2 > closeToInfty:
                    self.assertGreater(c1, closeToInfty)
                else:
                    self.assertLess(np.abs(c1-c2)/(c1+c2), 1e-2)  # 1% error in closed-loop H2 gain
                

        return

    # Need to make the below for Hinf somehow
    # def test_hinf(self):
    #     for matType in [np.matrix, np.array]:
    #         A = matType([[1,2],[0,3]])
    #         B = matType([[0,1]]).T
    #         Bdist = matType([[0,1]]).T
    #         C1 = matType([[1,0],[0,1],[0,0]])
    #         D12= matType([[0,0,1]]).T
    #
    #         KLMI = synth_h2_state_feedback_LMI(A, B, Bdist, C1, D12)
    #         K2, X2, J2 = controlpy.synthesis.controller_H2_state_feedback(A, B, Bdist, C1, D12)
    #
    #         self.assertLess(np.linalg.norm(K2-KLMI), 1e-3*np.linalg.norm(K2))
    #
    #     for nStates in [1,2,4,8]:
    #         for nInput in [1,3,5]:
    #             A = np.random.normal(size=[nStates,nStates])
    #             B = np.random.normal(size=[nStates,nInput])
    #             Bdist = np.random.normal(size=[nStates,nInput])
    #             nOut = nInput + nStates
    #             C1 = np.random.normal(size=[nOut, nStates])
    #             C1[nStates:,:] = 0
    #             D12= np.random.normal(size=[nOut, nInput])
    #             D12[:nStates,:] = 0
    #
    #             KLMI = synth_h2_state_feedback_LMI(A, B, Bdist, C1, D12)
    #             K2, X2, J2 = controlpy.synthesis.controller_H2_state_feedback(A, B, Bdist, C1, D12)
    #
    #             # print(KLMI)
    #             # print(K2)
    #
    #             c1 = controlpy.analysis.system_norm_H2(A-B@KLMI, Bdist, C1)
    #             c2 = controlpy.analysis.system_norm_H2(A-B@K2, Bdist, C1)
    #
    #             print(c1,c2)
    #
    #             closeToInfty = 100
    #             if c1 > closeToInfty:
    #                 self.assertGreater(c2, closeToInfty)
    #             elif c2 > closeToInfty:
    #                 self.assertGreater(c1, closeToInfty)
    #             else:
    #                 self.assertLess(np.abs(c1-c2)/(c1+c2), 1e-2)  # 1% error in closed-loop H2 gain
    #
    #
    #     return

            



if __name__ == '__main__':
    np.random.seed(1)
    unittest.main()
