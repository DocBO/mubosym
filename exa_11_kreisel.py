# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 23:07:13 2015

@author: oliver
"""

import numpy as np
from sympy import symbols
import mubosym as mbs


###############################################################
# general system setup example
myMBS = mbs.MBSworld('kreisel', connect=True, force_db_setup=False)

#prepare a standard 
I = [50.,50.,50.]
I0 = [1.,1.,1.]
###################################
# Kreisel reloaded
#myMBS.add_body_3d('B1', 'world_M0', 1.0, I , 'free-3-rotate', parameters = []) #[np.pi/2., 2.0])
#myMBS.add_marker('B1_M', 'B1', 0., 0., 0., 0., 0., 0.)
#
#myMBS.add_body_3d('B2', 'B1_M', 10.0, I0, 'rod-zero', parameters = [2.0, 'X'])
#myMBS.add_force_special('B2', 'grav')
#
#x0 = np.hstack(( 0. * np.zeros(myMBS.dof), 0. * np.ones(myMBS.dof)))
#x0[3] = 30.

myMBS.add_body_3d('B1', 'world_M0', 1.0, I0 , 'free-3-rotate', parameters = []) #[np.pi/2., 2.0])
myMBS.add_marker('B1_M', 'B1', 0., 0., 0., 0., 0., 0.)
myMBS.add_body_3d('B2', 'B1_M', 10.0, I0, 'rod-zero', parameters = [2.0, 'X'])
myMBS.add_marker('B2_M', 'B2', 0., 0., 0., 0., 0., 0.)
myMBS.add_force_special('B2', 'grav')
myMBS.add_body_3d('B3', 'B2_M', 1.0, I, 'revolute', parameters = ['X'])
x0 = np.hstack(( 0. * np.zeros(myMBS.dof), 0. * np.ones(myMBS.dof)))
x0[7] = 100.


#################################################
# parameters
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)


#################################################
# external force definitions
#def ext_sinus(t):
#    omega = 0.5*t
#    return 5000.0*math.sin(omega * t)
#    
#def null(t):
#    return 0.
#
#myMBS.add_force_ext('tire', 'world_M0', 0.,1.,0., ext_sinus)


myMBS.kaneify()

moving_frames_in_graphics = ['B3']
fixed_frames_in_graphics = []

myMBS.prep_lambdas(moving_frames_in_graphics, fixed_frames_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 15.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e+0)

x_final = myMBS.x_t[-1]
################################################
# linear analysis of the last state (returns also the jacobian)
jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)

myMBS.prepare(save=False)
myMBS.animate(t_max, dt, scale = 4, time_scale = 1, t_ani = 15, labels = True, plots='standard')