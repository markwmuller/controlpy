from __future__ import print_function, division

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

np.random.seed(1234)

import controlpy

import unittest

class TestIsHurwitz(unittest.TestCase):

    def test_scalar_hurwitz(self):
        self.assertTrue(controlpy.analysis.is_hurwitz(np.matrix([[-1]])))
        self.assertTrue(controlpy.analysis.is_hurwitz(np.array([[-1]])))
        self.assertFalse(controlpy.analysis.is_hurwitz(np.matrix([[1]])))
        self.assertFalse(controlpy.analysis.is_hurwitz(np.array([[1]])))
          
 
    def test_multidim_hurwitz(self):
        for matType in [np.matrix, np.array]:
            Astable = matType([[-1,2,1000], [0,-2,3], [0,0,-9]])
            Aunstable = matType([[1,2,1000], [0,-2,3], [0,0,-9]])
            Acritical = matType([[0,2,10], [0,-2,3], [0,0,-9]])
              
            tolerance = -1e-6 # for critically stable case
              
            for i in range(1000):
                T = matType(np.random.normal(size=[3,3]))
                Tinv = np.linalg.inv(T)
                  
                Bs = np.dot(np.dot(T,Astable),Tinv)
                Bu = np.dot(np.dot(T,Aunstable),Tinv)
                Bc = np.dot(np.dot(T,Acritical),Tinv)
  
                self.assertTrue(controlpy.analysis.is_hurwitz(Bs), str(i))
                self.assertFalse(controlpy.analysis.is_hurwitz(Bu), str(i))
                  
                self.assertFalse(controlpy.analysis.is_hurwitz(Bc, tolerance=tolerance), 'XXX'+str(i)+'_'+str(np.max(np.real(np.linalg.eigvals(Bc)))))
         
     
#     def test_uncontrollable_modes(self):
#         #TODO: FIXME: THIS FAILS!
#         for matType in [np.matrix, np.array]:
#             
#             es = [-1,0,1]
#             for e in es:
#                 A = matType([[e,2,0],[0,e,0],[0,0,e]])
#                 B = matType([[0,1,0]]).T
# 
#                 tolerance = 1e-9
#                 for i in range(1000):
#                     T = matType(np.random.normal(size=[3,3]))
#                     Tinv = np.linalg.inv(T)
#                      
#                     AA = np.dot(np.dot(T,A),Tinv)
#                     BB = np.dot(T,B)
#                      
#                     isControllable = controlpy.analysis.is_controllable(AA, BB, tolerance=tolerance)
#                     self.assertFalse(isControllable)
#                     
#                     isStabilisable = controlpy.analysis.is_stabilisable(A, B)
#                     self.assertEqual(isStabilisable, e<0)
# 
#                     if 0:
#                         #These shouldn't fail!
#                         uncontrollableModes, uncontrollableEigenValues = controlpy.analysis.uncontrollable_modes(AA, BB, returnEigenValues=True, tolerance=tolerance)
#                         self.assertEqual(len(uncontrollableModes), 1)
#           
#                         self.assertAlmostEqual(uncontrollableEigenValues[0], 1, delta=tolerance)
#                         self.assertAlmostEqual(np.linalg.norm(uncontrollableModes[0] - np.matrix([[0,0,1]]).T), 0, delta=tolerance)
                     
                     
     
    def test_time_discretisation(self):
        for matType in [np.matrix, np.array]:
            Ac = matType([[0,1],[0,0]])
            Bc = matType([[0],[1]])
             
            dt = 0.1
             
            Ad = matType([[1,dt],[0,1]])
            Bd = matType([[dt**2/2],[dt]])
            for i in range(1000):
                T = matType(np.random.normal(size=[2,2]))
                Tinv = np.linalg.inv(T)
 
                AAc = np.dot(np.dot(T,Ac),Tinv)
                BBc = np.dot(T,Bc)
 
                AAd = np.dot(np.dot(T,Ad),Tinv)
                BBd = np.dot(T,Bd)
                 
                AAd2, BBd2 = controlpy.analysis.discretise_time(AAc, BBc, dt)
                 
                self.assertLess(np.linalg.norm(AAd-AAd2), 1e-6)
                self.assertLess(np.linalg.norm(BBd-BBd2), 1e-6)
                 
                 
            #test some random systems against Euler discretisation
            nx = 20
            nu = 20
            dt = 1e-6
            tol = 1e-3
            for i in range(1000):
                Ac = matType(np.random.normal(size=[nx,nx]))
                Bc = matType(np.random.normal(size=[nx,nu]))
                 
                Ad1 = np.identity(nx)+Ac*dt
                Bd1 = Bc*dt
                 
                Ad2, Bd2 = controlpy.analysis.discretise_time(Ac, Bc, dt)
                 
                self.assertLess(np.linalg.norm(Ad1-Ad2), tol)
                self.assertLess(np.linalg.norm(Bd1-Bd2), tol)
                 
            

    def test_system_norms(self):
        for matType in [np.matrix, np.array]:
            A = matType([[-1,2],[0,-3]])
            B = matType([[0,1]]).T
            C = matType([[1,0]])
            
            h2norm = controlpy.analysis.system_norm_H2(A, B, C)
            hinfnorm = controlpy.analysis.system_norm_Hinf(A, B, C)
            
            h2norm_sl = 0.4082483
            hinfnorm_sl = 0.6666667

            tol = 1e-3
            self.assertLess(np.linalg.norm(h2norm-h2norm_sl), tol)
            self.assertLess(np.linalg.norm(hinfnorm-hinfnorm_sl), tol)
           
    #TODO:
    # - observability tests, similar to controllability
    



if __name__ == '__main__':
    np.random.seed(1)
    unittest.main()
