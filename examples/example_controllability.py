'''Example showing how to test a system's controllability
'''

from __future__ import print_function, division

import controlpy

import numpy as np

# A single input system, one uncontrollable mode.
A = np.matrix([[0,1,0],[0,0,1],[0,0,5]])
B = np.matrix([[0],[1],[0]])

uncontrollableModes = controlpy.analysis.uncontrollable_modes(A,B)

if not uncontrollableModes:
    print('System is controllable.')
else: 
    print('System is uncontrollable. Uncontrollable modes are:')
    print(uncontrollableModes)







