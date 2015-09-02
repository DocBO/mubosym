# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:54:16 2015

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
myMBS = mbs.MBSworld('new_constraint', connect=True, force_db_setup=False)


#prepare a standard 
I = [0.,0.,0.]

######################################
# choose
setup = 'one-dof'
setup = 'two-dof'

if setup == 'two-dof':
    myMBS.add_marker('world_M1','world', 0.,0.,0., 0., 0.,0.)
    myMBS.add_body_3d('b1','world_M1', 1.0, I, 'xy-plane', parameters = [])
    myMBS.add_force_special('b1', 'grav')
    x0 = np.hstack((1.,0.,0.,0.))
#
elif setup == 'one-dof':
    myMBS.add_marker('world_M1','world', 0.,0.,0., 0., 0.,0.)
    myMBS.add_body_3d('b1','world_M1', 1.0, I, 'rod-1-cardanic-efficient', parameters = [1,0,0.])
    myMBS.add_force_special('b1', 'grav')
    x0 = np.hstack(( np.pi/2. * np.ones(myMBS.dof), 0. * np.ones(myMBS.dof)))
else:
    exit(0)


factor = 10.

R = myMBS.get_frame('world_M0')
x = R[0]
y = R[1]
z = R[2]
   
if setup == 'two-dof':
    equ1 = x**2 + y**2 - 1
    myMBS.add_geometric_constaint('b1', equ1, 'world_M0', factor)


#################################################
# constants
g = symbols('g')
constants = [ g ]           # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)


myMBS.kaneify()

fixed_frames_in_graphics = ['world_M1']
frames_in_graphics = ['b1']
forces_in_graphics = ['b1']
myMBS.prep_lambdas(frames_in_graphics, fixed_frames_in_graphics, forces_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 10.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e+0)

myMBS.prepare(DATA_PATH, save=True)
myMBS.animate(t_max, dt, scale = 4, time_scale = 0.5, t_ani = 10.)
