'''A simple example script, compute the norm of an LTI system
'''

from __future__ import print_function, division

import controlpy

import numpy as np

# The system is 
# dx = A*x + B*u
# z = C*x 
# u is a disturbance.

A = np.matrix([[-0.1, 10],[0,-0.7]])
B = np.matrix([[0],[1]])
C = np.matrix([[1,0]])

print('H2 norm: ',controlpy.analysis.system_norm_H2(A, B, C))
print('Hinf norm: ',controlpy.analysis.system_norm_Hinf(A, B, C, precision=1e-6))
