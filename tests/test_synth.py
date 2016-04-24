from __future__ import print_function, division

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

np.random.seed(1234)

import controlpy

import unittest

tolerance = 1e-4

class TestSynthesis(unittest.TestCase):

    def test_h2_hinf(self):
        for matType in [np.matrix, np.array]:
            A = matType([[1,2],[0,3]])
            B = matType([[0,1]]).T
            Bdist = matType([[0,1]]).T
            C1 = matType([[1,0],[0,1],[0,0]])
            D12= matType([[0,0,1]]).T
            #scilab results:
            K2sl = matType([[  - 4.52044  ,- 8.2992226  ]])
            X2sl = matType([[9.7171888,    4.52044],[ 4.52044,     8.2992226  ]])
            
            K2, X2, J2 = controlpy.synthesis.controller_H2_state_feedback(A, B, Bdist, C1, D12, useLMI=False)
            
            self.assertLess(np.linalg.norm(K2+K2sl), tolerance) #note sign difference
            self.assertLess(np.linalg.norm(X2-X2sl), tolerance) 

            K2LMI, _, _ = controlpy.synthesis.controller_H2_state_feedback(A, B, Bdist, C1, D12, useLMI=True)

            self.assertLess(np.linalg.norm(K2-K2LMI), tolerance) #note sign difference

            
            
#             Kinf, Xinf, Jinf = controlpy.synthesis.controller_Hinf_state_feedback(A, B, Bdist, C1, D12, subOptimality=1.1)
#             print(Kinf)
#             print(Xinf)
#             print(Jinf)



if __name__ == '__main__':
    np.random.seed(1)
    unittest.main()
