# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 23:07:13 2015

@author: oliver
"""
import numpy as np
from sympy import symbols, sin
import mubosym as mbs

###############################################################
# general system setup example
myMBS = mbs.MBSworld('bending_stiffness', connect=True, force_db_setup=False)

#prepare a standard 
I = [10.,10.,10.]
###################################
# a complex torsional and bending stiffness example
myMBS.add_marker('world_M1', 'world', 0.,0.,1.)
myMBS.add_body_3d('rod_1', 'world_M1', 1.0, I , 'rod-1-cardanic', parameters = [0.,0.]) #[np.pi/2., 2.0])
myMBS.add_torque_3d('rod_1', 'bending-stiffness-1', parameters=[np.pi,800.])# [0.,0.,0.]])
myMBS.add_force_special('rod_1', 'grav')
myMBS.add_marker('rod_1_M0', 'rod_1', 0.,0.,0.)

myMBS.add_body_3d('rod_2', 'rod_1_M0', 1.0, I, 'rod-1-cardanic', parameters = [-1.0,np.pi/2.])
myMBS.add_torque_3d('rod_2', 'bending-stiffness-1', parameters=[0.,800.])
myMBS.add_force_special('rod_2', 'grav')
myMBS.add_marker('rod_2_M0', 'rod_2', 0.,0.,0.)
#

myMBS.add_body_3d('rod_3', 'rod_2_M0', 1.0, I, 'rod-zero-Z', parameters = [2.0])
myMBS.add_force_special('rod_3', 'grav')

myMBS.add_marker('rod_3_M0', 'rod_3', 0.,0.,0.)
#
#
myMBS.add_body_3d('rod_4', 'rod_3_M0', 1.0, I, 'rod-1-revolute', parameters = [-0.5,0.,2.0])
myMBS.add_force_special('rod_4', 'grav')
myMBS.add_torque_3d('rod_4', 'rotation-stiffness-1', parameters = [100.])

#
x0 = np.hstack(( np.pi / 3. * np.ones(myMBS.dof), 1.0 * np.ones(myMBS.dof)))


 
for b in myMBS.bodies.keys():
    myMBS.add_damping(b,5.0)
    


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

moving_frames_in_graphics = ['rod_1','rod_2','rod_3','rod_4']
fixed_frames_in_graphics = []
myMBS.prep_lambdas(moving_frames_in_graphics, fixed_frames_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 30.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e-0)

x_final = myMBS.x_t[-1]
################################################
# linear analysis of the last state (returns also the jacobian)
jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)

myMBS.prepare(mbs.DATA_PATH, save=True)
#myMBS.animate(t_max, dt, scale = 4, time_scale = 1, t_ani = 30.0, labels = True, plots='standard')
#myMBS.show_figures(t_max, dt)
myMBS.animate(t_max, dt, scale = 4, time_scale = 1.0, t_ani = 20.0)