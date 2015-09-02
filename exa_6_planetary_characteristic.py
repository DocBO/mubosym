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
myMBS = mbs.MBSworld('planetary_char', connect=True, force_db_setup=False)

#prepare a standard 
I = [1.,1.,1.]

##################################################
# kennlinie test for a planetary setup
#myMBS.add_marker(999, 0.,0.,1.)
myMBS.add_body_3d('sun', 'world_M0', 1000.0, I , 'xz-plane', parameters = [], graphics = False) #[np.pi/2., 2.0])
myMBS.add_marker('sun_M1', 'sun', 0.,0.,0.)
myMBS.add_force_special('sun', 'grav')

myMBS.add_body_3d('planet1','sun_M1', 10.0, I , 'xz-plane', parameters = [], graphics = False) #[np.pi/2., 2.0])
myMBS.add_body_3d('planet2','sun_M1', 10.0, I , 'xz-plane', parameters = [], graphics = False)


myMBS.add_force_spline_r('sun','planet1', DATA_PATH+'/force_kl1.dat', [0., -1.0])
myMBS.add_force_spline_r('sun','planet2', DATA_PATH+'/force_kl1.dat', [0., -1.0])


x0 = np.hstack(( 0.,0.,1.,1.,-1.,-1., 0.,0.,1.,0.,0.,1.))
#x0 = np.array([ -3.63235000e-01,  -6.54750000e-01,  0.,0.])
         
body_frames_in_graphics = ['sun','planet1','planet2']
fixed_frames_in_graphics = []


#for b in myMBS.bodies.keys():
#    myMBS.add_damping(b,0.1)
    
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
t_max = 40.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0)

#get the final state 
x_final = myMBS.x_t[-1]
################################################

myMBS.prepare(DATA_PATH, save=True)
myMBS.animate(t_max, dt, scale = 4, time_scale = 1, t_ani = 20.0)
