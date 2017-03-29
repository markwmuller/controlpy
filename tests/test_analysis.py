from __future__ import print_function, division

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

np.random.seed(1234)

import controlpy

import unittest

import cvxpy
def sys_norm_h2_LMI(Acl, Bdisturbance, C):
    #doesn't work very well, if problem poorly scaled Riccati works better.
    #Dullerud p 210
    n = Acl.shape[0]
    X = cvxpy.Semidef(n)
    Y = cvxpy.Semidef(n)

    constraints = [ Acl*X + X*Acl.T + Bdisturbance*Bdisturbance.T == -Y,
                  ]

    obj = cvxpy.Minimize(cvxpy.trace(Y))

    prob = cvxpy.Problem(obj, constraints)
    
    prob.solve()
    eps = 1e-16
    if np.max(np.linalg.eigvals((-Acl*X - X*Acl.T - Bdisturbance*Bdisturbance.T).value)) > -eps:
        print('Acl*X + X*Acl.T +Bdisturbance*Bdisturbance.T is not neg def.')
        return np.Inf

    if np.min(np.linalg.eigvals(X.value)) < eps:
        print('X is not pos def.')
        return np.Inf

    return np.sqrt(np.trace(C*X.value*C.T))


def sys_norm_hinf_LMI(A, Bdisturbance, C, D = None):
    '''Compute a system's Hinfinity norm, using an LMI approach.
    
    Acl, Bdisturbance are system matrices, describing the systems dynamics:
     dx/dt = Acl*x  + Bdisturbance*v
    where x is the system state and v is the disturbance.
    
    The system output is:
     z = C*x + D*v
    
    The matrix Acl must be Hurwitz for the Hinf norm to be finite. 
    
    Parameters
    ----------
    A  : (n, n) Matrix
         Input
    Bdisturbance : (n, m) Matrix
         Input
    C : (q, n) Matrix
         Input
    D : (q,m) Matrix
         Input (optional)

    Returns
    -------
    Jinf : Systems Hinf norm.
    
    See: Robust Controller Design By Convex Optimization, Alireza Karimi Laboratoire d'Automatique, EPFL
    '''
    
    if not controlpy.analysis.is_hurwitz(A):
        return np.Inf

    n = A.shape[0]
    ndist = Bdisturbance.shape[1]
    nout  = C.shape[0]

    X = cvxpy.Semidef(n)
    g = cvxpy.Variable()
    
    if D is None:
        D = np.matrix(np.zeros([nout, ndist]))
        
    r1 = cvxpy.hstack(cvxpy.hstack(A.T*X+X*A, X*Bdisturbance), C.T)
    r2 = cvxpy.hstack(cvxpy.hstack(Bdisturbance.T*X, -g*np.matrix(np.identity(ndist))), D.T)
    r3 = cvxpy.hstack(cvxpy.hstack(C, D), -g*np.matrix(np.identity(nout)))
    tmp = cvxpy.vstack(cvxpy.vstack(r1,r2),r3)
                        
    constraints = [tmp == -cvxpy.Semidef(n + ndist + nout)]

    obj = cvxpy.Minimize(g)

    prob = cvxpy.Problem(obj, constraints)
    
    try:
        prob.solve()#solver='CVXOPT', kktsolver='robust')
    except cvxpy.error.SolverError:
        print('Solution not found!')
        return None
    
    if not prob.status == cvxpy.OPTIMAL:
        return None
    
    return g.value



class TestAnalysis(unittest.TestCase):

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
                 
            

    def test_system_norm_H2(self):
        for matType in [np.matrix, np.array]:
            A = matType([[-1,2],[0,-3]])
            B = matType([[0,1]]).T
            C = matType([[1,0]])
            
            h2norm = controlpy.analysis.system_norm_H2(A, B, C)
            
            h2norm_lmi = sys_norm_h2_LMI(A, B, C)

            tol = 1e-3
            self.assertLess(np.linalg.norm(h2norm-h2norm_lmi), tol)
            

    def test_system_norm_Hinf(self):
        for matType in [np.matrix, np.array]:
            A = matType([[-1,2],[0,-3]])
            B = matType([[0,1]]).T
            C = matType([[1,0]])
            
            hinfnorm = controlpy.analysis.system_norm_Hinf(A, B, C)
            
            hinfnorm_lmi = sys_norm_hinf_LMI(A, B, C)

            tol = 1e-3
            self.assertLess(np.linalg.norm(hinfnorm-hinfnorm_lmi), tol)
           
    #TODO:
    # - observability tests, similar to controllability
    



if __name__ == '__main__':
    np.random.seed(1)
    unittest.main()
