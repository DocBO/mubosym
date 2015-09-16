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
myMBS = mbs.MBSworld('harmonic_oscillator', connect=False, force_db_setup=False)

#prepare a standard 
I = [1.,1.,1.]

##################################################
# kennlinie test for a planetary setup
#myMBS.add_marker(999, 0.,0.,1.)
myMBS.add_body_3d('atom1', 'world_M0', 10.0, I , 'xz-plane', parameters = [], graphics = False) #[np.pi/2., 2.0])
myMBS.add_marker('atom1_M', 'atom1', 0.,0.,0.)
myMBS.add_body_3d('atom2','world_M0', 10.0, I , 'xz-plane', parameters = [], graphics = False) #[np.pi/2., 2.0])


myMBS.add_two_body_force_model('harmonic_oscillator', 'atom1','atom2', 'harmonic', [1000.,1.1,8.])
#myMBS.add_force_spline_r('sun','planet2', mbs.DATA_PATH+'/force_kl1.dat', [0., -1.0])

#x0 = np.hstack(( 0. * np.ones(myMBS.dof), 0. * np.ones(myMBS.dof)))
#x0 = np.hstack(( 0.,0.,1.,1.))
x0 = np.array([ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  -1.])
         
body_frames_in_graphics = ['atom1','atom2']
fixed_frames_in_graphics = []

#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)

#for b in myMBS.bodies.keys():
#    myMBS.add_damping(b,2.0)

myMBS.kaneify()
myMBS.prep_lambdas(body_frames_in_graphics, fixed_frames_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 10.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0)

#get the final state 
x_final = myMBS.x_t[-1]
################################################

myMBS.prepare(mbs.DATA_PATH, save=False)
#myMBS.plotting(t_max, dt, plots='standard')
myMBS.animate(t_max, dt, scale = 4, time_scale = 1, t_ani = 10.0)
