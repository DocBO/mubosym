# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:15:49 2015

@author: oliver
"""
import numpy as np
from sympy import symbols
import mubosym as mbs



###############################################################
# general system setup example
myMBS = mbs.MBSworld('rotating_pendulum', connect=False, force_db_setup=False)

I = [1.,1.,1.]

#############################################################
# rotating frame constraint
#
omega = 1.0
A = 5.5
def rotation_inp(t):
    return A*t #np.sin(omega*t)
    
def rotation_inp_diff(t):
    return A #*omega*np.cos(omega*t)

def rotation_inp_diff_2(t):
    return 0.# A*omega*omega*np.sin(omega*t)

#use of an external parameter (be carefull: you have to provide also the time derivatives up to second order)
myMBS.add_parameter('phi', rotation_inp, rotation_inp_diff, rotation_inp_diff_2)
myMBS.add_rotating_marker_para('rot_M0', 'world', 'phi', 0., 1.5, 0., 'Y')

#alternatively use of a steady rotating frame without extra parameter name, constant omega
#myMBS.add_rotating_marker('rot_M0', 'world',0., 0., 0., 2.5, 'Y')

myMBS.add_body_3d('mount', 'rot_M0', 1.0, I , 'rod-zero-X', parameters = [1.0]) #[np.pi/2., 2.0])
myMBS.add_force_special('mount', 'grav')


myMBS.add_marker('mount_M0', 'mount', 0.,0.,0.)
myMBS.add_body_3d('pendulum', 'mount_M0', 1.0, I , 'rod-1-cardanic', parameters = [-1.5,0.]) #[np.pi/2., 2.0])
myMBS.add_force_special('pendulum', 'grav')

x0 = np.hstack(( 0. * np.ones(myMBS.dof), 0. * np.zeros(myMBS.dof)))

  
body_frames_in_graphics = ['mount', 'pendulum']
fixed_frames_in_graphics = []


for b in myMBS.bodies.keys():
    myMBS.add_damping(b,0.5)
    


#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)


myMBS.kaneify()
myMBS.prep_lambdas(body_frames_in_graphics, fixed_frames_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 30.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0)

x_final = myMBS.x_t[-1]
################################################
# linear analysis of the last state (returns also the jacobian)
jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)

myMBS.prepare(mbs.DATA_PATH, save=True)
myMBS.animate(t_max, dt, scale = 4, time_scale = 1.0, t_ani = 20., labels = True)
