# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:41:09 2015

@author: oliver
"""
import os, sys
import numpy as np
from sympy import symbols
BASE_PATH = os.path.dirname( os.path.realpath ( __file__) )
DATA_PATH = BASE_PATH + '/data'
sys.path.append(BASE_PATH+"/mubosym") #python 3 compatibility (later on)

import mubosym as mbs
mbs.BASE_PATH = BASE_PATH


###############################################################
# general system setup example
myMBS = mbs.MBSworld('swing_table', connect=True, force_db_setup=False)


#prepare a standard 
I = [0.,0.,0.]

######################################
# some loop constraint
myMBS.add_marker('world_M1','world', 1.,0.,0.)
myMBS.add_body_3d('b1','world_M0', 1.0, I, 'rod-1-cardanic', parameters = [1.0,0.])
myMBS.add_force_special('b1', 'grav')
#
myMBS.add_marker('b1_M0','b1', 0.,0.,0.)
myMBS.add_body_3d('b2', 'b1_M0', 1.0, I, 'rod-1-cardanic', parameters = [1.5,0.])
myMBS.add_force_special('b2', 'grav')

myMBS.add_marker('b2_M0','b2', 0.,0.,0.)
myMBS.add_body_3d('b3', 'b2_M0', 1.0, I, 'rod-1-cardanic', parameters = [1.0,0.])
myMBS.add_force_special('b3', 'grav')


#x0 = np.hstack(( 0. * np.ones(myMBS.dof), 0. * np.ones(myMBS.dof)))
x0 = np.array([-13.73437514,  15.30494211,  -5.87985813,   0.81904538,
        -0.81998223,   0.82150335])
frames_in_graphics = ['b1', 'b2']

factor = 5.

R = myMBS.get_frame('world_M0')
x = R[0]
y = R[1]
z = R[2]

# say: x - 0 = 0 in IF
#IF = myMBS.body_frames[999]


for b in myMBS.bodies.keys():
    myMBS.add_damping(b,0.1)
    

equ1 = y
myMBS.add_geometric_constaint('b3', equ1, 'world_M0', factor)
equ2 = x-1.5
myMBS.add_geometric_constaint('b3', equ2, 'world_M0', factor)

#another constraint
#equ2 = x
#myMBS.add_geometric_constaint(2, equ2, IF, factor, mySyms)

#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)


#for ii in range(len(myMBS.eq_constr)/2):
#    x0 = myMBS.correct_the_initial_state(myMBS.n_constr[ii], x0)


myMBS.kaneify()
myMBS.prep_lambdas(frames_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 20.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0)

x_final = myMBS.x_t[-1]


x00 = dict(zip(myMBS.q_flat+myMBS.u_flat, x_final))


x_op = {myMBS.q_flat[0]: 0.0, myMBS.q_flat[1]: np.pi/2., myMBS.q_flat[2]: np.pi/2., myMBS.u_flat[1]: 0.0, myMBS.u_flat[0]: 0.0, myMBS.u_flat[2]: 0.0}
#x_op = {myMBS.q_flat[0]: 0.0, myMBS.q_flat[1]: 0.0, myMBS.q_flat[2]: 0.0, myMBS.u_flat[1]: 0.0, myMBS.u_flat[0]: 0.0, myMBS.u_flat[2]: 0.0}

a_op = {myMBS.a_flat[0]: 0.0, myMBS.a_flat[1]: 0., myMBS.a_flat[2]: 0.}


jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)



myMBS.prepare(DATA_PATH, save=False)
myMBS.animate(t_max, dt, scale = 4, time_scale = 1, t_ani = 20.0)

myMBS.linearize(x_op, a_op)#, quad = True)