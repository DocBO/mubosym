# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:54:16 2015

@author: oliver
"""

import numpy as np
from sympy import symbols
import mubosym as mbs

###############################################################
# general system setup example
myMBS = mbs.MBSworld('crank_slider', connect=False, force_db_setup=False)


#prepare a standard 
I = [0.,0.,0.]

######################################
# some loop constraint
myMBS.add_marker('world_M1','world', 0.,0.,0., 0., 0.,0.)
#myMBS.add_body_3d('b1','world_M1', 1.0, I, 'rod-1-cardanic-efficient', parameters = [1.0,0.])
myMBS.add_body_3d('b1','world_M1', 1.0, I, 'rod-1-cardanic-efficient', parameters = [-1.0,0.])

myMBS.add_marker('b1_M0','b1', 0.,0.,0.)
myMBS.add_body_3d('b2', 'b1_M0', 1.0, I, 'rod-1-cardanic-efficient', parameters = [-3.0,0.])
myMBS.add_force_special('b2', 'grav')

######################################
# damping if needed
#for b in myMBS.bodies.keys():
#    myMBS.add_damping(b,2.5)
    

######################################
# start conditions
x0 = np.array([-14.75044927,   1.3777362 ,   5.4077404 ,   1.50199767])

######################################
# loop constraint
factor = 20.

R = myMBS.get_frame('world_M0')
x = R[0]
y = R[1]
z = R[2]

equ1 = y
myMBS.add_geometric_constaint('b2', equ1, 'world_M0', factor)


#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)

################################################
# signal example
vel_1 = myMBS.get_body('b1').get_vel_magnitude()
myMBS.add_control_signal(vel_1)

myMBS.kaneify()

fixed_frames_in_graphics = ['world_M1']
frames_in_graphics = ['b1', 'b2']
forces_in_graphics = ['b1', 'b2']
myMBS.prep_lambdas(frames_in_graphics, fixed_frames_in_graphics, forces_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 15.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e-0)

################################################
# jacobian example
jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)


myMBS.prepare(mbs.DATA_PATH, save=True)
################################################
# plotting example
#myMBS.plotting(t_max, dt, plots='standard')
################################################
# animation 
myMBS.animate(t_max, dt, scale = 4, time_scale = 1.0, t_ani = 5.)

################################################
# linearization example
#myMBS.linearize(x_op, a_op)#, quad = True)