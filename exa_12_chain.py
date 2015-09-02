# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:54:16 2015

@author: oliver
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:41:09 2015

@author: oliver
"""
import os, sys
import numpy as np
from sympy import symbols, sin

BASE_PATH = os.path.dirname( os.path.realpath ( __file__) )
DATA_PATH = BASE_PATH + '/data'
sys.path.append(BASE_PATH+"/mubosym") #python 3 compatibility (later on)

import mubosym as mbs
mbs.BASE_PATH = BASE_PATH

###############################################################
# general system setup example
myMBS = mbs.MBSworld('chain', connect=True, force_db_setup=False)

b_n = []
m_n = []
b_n_max = 3
for ii in range(b_n_max):
    b_n.append(str(ii))
    m_n.append(str(ii)+"_M0")


#prepare a standard 
I = [0.,0.,0.]

######################################
# large chain
#myMBS.add_marker('world_M1','world', 0.,0.,0., np.pi/2.0, 0.,0.)
myMBS.add_marker('world_M1','world', 0.,0.,0., 0., 0.,0.)
myMBS.add_body_3d(b_n[0],'world_M1', 1.0, I, 'rod-1-cardanic-efficient', parameters = [1.0,0.])
myMBS.add_force_special(b_n[0], 'grav')


for ii in range(0,b_n_max-1):
    #myMBS.add_marker(m_n[ii],b_n[0], (float(ii)+0.5)/2.0,0.,0.)
    myMBS.add_marker(m_n[ii],b_n[ii], 0.,0.,0.)
    myMBS.add_body_3d(b_n[ii+1], m_n[ii], 1.0, I, 'rod-1-cardanic-efficient', parameters = [1.0,0.])
    myMBS.add_force_special(b_n[ii+1], 'grav')


x0 = np.hstack(( 0. * np.ones(myMBS.dof), 4. * np.ones(myMBS.dof)))

if b_n_max == 10:
    x0 = np.array([  4.72435299e-01,  -5.70207174e+00,   7.43158372e-01,
         9.91607076e-01,   1.35451752e+00,   1.78215809e+00,
         2.14636522e+00,   2.39608001e+00,   2.55891136e+00,
         2.66809442e+00,   4.49071066e-05,  -9.07371463e-05,
         6.84665257e-05,  -3.12828392e-05,   1.16655539e-05,
        -8.76343033e-06,   2.10635295e-05,  -4.80622820e-05,
         6.54381274e-05,  -3.23797871e-05])
#x0[2] = 0.


factor = 5.
x,y,z = symbols('x y z')
mySyms = {'x':x,'y':y,'z':z}
# say: x - 0 = 0 in IF


#equ1 = y
#myMBS.add_geometric_constaint(b_n[-1], equ1, 'world_M0', factor, mySyms)
#equ2 = x - 7.0
#myMBS.add_geometric_constaint(b_n[-1], equ2, 'world_M0', factor, mySyms)

for b in myMBS.bodies.keys():
    myMBS.add_damping(b,0.4)
    

#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)


fixed_frames_in_graphics = ['world_M1']
frames_in_graphics = []
forces_in_graphics = []
myMBS.kaneify()
myMBS.prep_lambdas(frames_in_graphics, fixed_frames_in_graphics, forces_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 20.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e+0)




jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)


myMBS.prepare(DATA_PATH, save = True)
#myMBS.show_figures(t_max, dt)
myMBS.animate(t_max, dt, scale = 4, time_scale = 1.0, t_ani = 20.0)

#myMBS.linearize(x_op, a_op)#, quad = True)