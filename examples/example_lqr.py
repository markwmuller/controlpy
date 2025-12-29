'''A simple example script, which implements an LQR controller for a double integrator.
'''

from __future__ import print_function, division

import controlpy

import numpy as np

# Example system is a double integrator:
A = np.array([[0,1],[0,0]])
B = np.array([[0],[1]])

# Define our costs:
Q = np.array([[1,0],[0,0]])
R = np.array([[1]])

# Compute the LQR controller
gain, X, closedLoopEigVals = controlpy.synthesis.controller_lqr(A,B,Q,R)

print('The computed gain is:')
print(gain)

print('The closed loop eigenvalues are:')
print(closedLoopEigVals)





