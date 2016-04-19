import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

np.random.seed(1234)

import controlpy

import unittest

class TestIsHurwitz(unittest.TestCase):

#     def test_scalar_hurwitz(self):
#         self.assertTrue(controlpy.analysis.is_hurwitz(np.matrix([[-1]])))
#         self.assertTrue(controlpy.analysis.is_hurwitz(np.array([[-1]])))
#         self.assertFalse(controlpy.analysis.is_hurwitz(np.matrix([[1]])))
#         self.assertFalse(controlpy.analysis.is_hurwitz(np.array([[1]])))
#         
#
#     def test_multidim_hurwitz(self):
#         for matType in [np.matrix, np.array]:
#             Astable = matType([[-1,2,1000], [0,-2,3], [0,0,-9]])
#             Aunstable = matType([[1,2,1000], [0,-2,3], [0,0,-9]])
#             Acritical = matType([[0,2,10], [0,-2,3], [0,0,-9]])
#             
#             tolerance = -1e-6 # for critically stable case
#             
#             for i in range(1000):
#                 T = matType(np.random.normal(size=[3,3]))
#                 Tinv = np.linalg.inv(T)
#                 
#                 Bs = np.dot(np.dot(T,Astable),Tinv)
#                 Bu = np.dot(np.dot(T,Aunstable),Tinv)
#                 Bc = np.dot(np.dot(T,Acritical),Tinv)
# 
#                 self.assertTrue(controlpy.analysis.is_hurwitz(Bs), str(i))
#                 self.assertFalse(controlpy.analysis.is_hurwitz(Bu), str(i))
#                 
#                 self.assertFalse(controlpy.analysis.is_hurwitz(Bc, tolerance=tolerance), 'XXX'+str(i)+'_'+str(np.max(np.real(np.linalg.eigvals(Bc)))))
        
    
    def test_uncontrollable_modes(self):
        for matType in [np.matrix, np.array]:
            A = matType([[1,2,0],[0,1,0],[0,0,1]])
            B = matType([[0,1,0]]).T

            tolerance = 1e-9
            for i in range(1000):
                T = np.identity(3) # matType(np.random.normal(size=[3,3]))
                Tinv = np.linalg.inv(T)
                
                AA = np.dot(np.dot(T,A),Tinv)
                BB = np.dot(T,B)
                
                isControllable = controlpy.analysis.is_controllable(AA, BB, tolerance=tolerance)
                self.assertFalse(isControllable)
                uncontrollableModes, uncontrollableEigenValues = controlpy.analysis.uncontrollable_modes(AA, BB, returnEigenValues=True, tolerance=tolerance)
                self.assertEqual(len(uncontrollableModes), 1)

                self.assertAlmostEqual(uncontrollableEigenValues[0], 1, delta=tolerance)
                self.assertAlmostEqual(np.linalg.norm(uncontrollableModes[0] - np.matrix([[0,0,1]]).T), 0, delta=tolerance)
                
                



if __name__ == '__main__':
    unittest.main()
