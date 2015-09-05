# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:41:09 2015

@author: oliver
"""
import numpy as np
from sympy import symbols
import mubosym as mbs



###############################################################
# general system setup example
myMBS = mbs.MBSworld('strange_pendulum', connect=True, force_db_setup=False)

b_n = []
m_n = []
b_n_max = 3
for ii in range(b_n_max):
    b_n.append(str(ii))
    m_n.append(str(ii)+"_M0")


I = [1.,1.,1.]

##################################
# cracy pendulum
myMBS.add_marker('world_M1', 'world',0.,0.,-np.pi/4.,0.,0.) #np.pi/4.,np.pi/4.)
#b_n[0] = myMBS.add_body_3d(999, 1, 1.0, I , 'rod-revolute', parameters = [0.,np.pi/2,2.0]) #[np.pi/2., 2.0])
#b_n[0] = myMBS.add_body_3d(999, 1, 1.0, I , 'rod-2-cardanic', parameters = [2.0]) #[np.pi/2., 2.0])
myMBS.add_body_3d(b_n[0], 'world_M1', 1.0, I , 'angle-rod', parameters = [np.pi/4., 2.0])
myMBS.add_force_special(b_n[0], 'grav')

myMBS.add_marker(m_n[0], b_n[0], 0.,0.,0.)
myMBS.add_body_3d(b_n[1], m_n[0], 1.0, I, 'angle-rod', parameters = [np.pi/4., 2.0])
myMBS.add_force_special(b_n[1], 'grav')

myMBS.add_marker(m_n[1], b_n[1], 0.,0.,0.)
myMBS.add_body_3d(b_n[2], m_n[1], 1.0, I, 'angle-rod', parameters = [np.pi/4., 2.0])
myMBS.add_force_special(b_n[2], 'grav')

#myMBS.add_marker(b_n[2], 0.,0.,0.)
#b_n[3] = myMBS.add_body_3d(b_n[2], 0, 1.0, I, 'angle-rod', parameters = [np.pi/4., 2.0])
#myMBS.add_force(b_n[3], 'grav')




x0 = np.hstack(( 1. * np.ones(myMBS.dof), 1. * np.ones(myMBS.dof)))


#for b in myMBS.bodies.keys():
#    myMBS.add_damping(b,0.1)
    


#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)



body_frames_in_graphics = [b_n[0],b_n[1],b_n[2]]
fixed_frames_in_graphics = []

myMBS.kaneify()
myMBS.prep_lambdas(body_frames_in_graphics, fixed_frames_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 30.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0)


jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)

myMBS.prepare(mbs.DATA_PATH, save=True)
myMBS.animate(t_max, dt, scale = 4, time_scale = 1, t_ani = 30.0)
