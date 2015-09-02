# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:15:49 2015

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

############################################################
# general system setup example
myMBS = mbs.MBSworld('moving_pendulum', connect=True, force_db_setup=False)

I = [0.,0.,0.]

############################################################
# rotating frame constraint
#
omega = 2.5 #try up to 30
A = 2.0
def rotation_inp(t):
    return A*np.sin(omega*t)
    
def rotation_inp_diff(t):
    return A*omega*np.cos(omega*t)

def rotation_inp_diff_2(t):
    return -A*omega*omega*np.sin(omega*t)

myMBS.add_parameter('phi', rotation_inp, rotation_inp_diff, rotation_inp_diff_2)
myMBS.add_moving_marker_para('rot_M0', 'world', 'phi', 0., 0., 0., 'X')


#myMBS.add_body_3d('mount', 'rot_M0', 1.0, I , 'rod-zero', parameters = [1.0,'X']) #[np.pi/2., 2.0])
#myMBS.add_force_special('mount', 'grav')


#myMBS.add_marker('mount_M0', 'mount', 0.,0.,0.)
myMBS.add_body_3d('pendulum', 'rot_M0', 1.0, I , 'rod-1-cardanic', parameters = [1.5,0.]) #[np.pi/2., 2.0])
myMBS.add_force_special('pendulum', 'grav')

x0 = np.hstack(( 0. * np.ones(myMBS.dof), 0. * np.ones(myMBS.dof)))

  



for b in myMBS.bodies.keys():
    myMBS.add_damping(b,0.05)
    


#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)


myMBS.kaneify()

body_frames_in_graphics = ['rot_M0','pendulum']
fixed_frames_in_graphics = []
bodies_in_graphics = {'pendulum': 'sphere'}

myMBS.prep_lambdas(body_frames_in_graphics, fixed_frames_in_graphics, [], bodies_in_graphics)


dt = 0.01  # refine if necesarry
t_max = 20.
####################

####
myMBS.inte_grate_full(x0, t_max, dt, mode = 0)

x_final = myMBS.x_t[-1]
################################################
# linear analysis of the last state (returns also the jacobian)
jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)

myMBS.prepare(DATA_PATH, save=True)

#use a smaller time scale for good animation results
myMBS.animate(t_max, dt, scale = 4, time_scale = 1.0, t_ani = 20.)
