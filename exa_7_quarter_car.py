# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:41:09 2015

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
myMBS = mbs.MBSworld('quarter_car', connect=True, force_db_setup=False)

#prepare a standard 
I_car = [500.,500.,500.]
I = [0.,0.,0.]
I_tire = [1.,1.,1.]
###################################
# quarter car model - revisited

myMBS.add_marker('world_M1', 'world', 0.,0.,0.)


###################################
# body of car
myMBS.add_body_3d('car_body', 'world_M1', 500.0, I_car , 'free-3-translate-z-rotate', parameters = [], graphics = False)
myMBS.add_marker('car_body_M0', 'car_body', 0.,0.,0.)
myMBS.add_force_special('car_body', 'grav')

myMBS.add_body_3d('tire_carrier', 'car_body_M0', 50.0, I , 'y-axes', parameters = [])
###################################
# body of tire


A = -0.15
omega = 0.8
def rotation_inp(t):
    return A*np.sin(omega*t)
    
def rotation_inp_diff(t):
    return A*omega*np.cos(omega*t)

def rotation_inp_diff_2(t):
    return -A*omega*omega*np.sin(omega*t)

def rotation_inp_expr():
    t, A = symbols('t A')
    return A*sin(1.0*t)

k = mbs.interp1d_interface.interp(filename = DATA_PATH+'/vel_01.dat')
#high end definition of static variables...
@mbs.static_vars(t_p=0, diff_p=0)
def lateral_inp(t): 
    #return -20.
    velocity = myMBS.get_control_signal(0)
    v_soll = k.f_interp(t)
    diff = (v_soll-velocity)/10.0
    delt = (t-lateral_inp.t_p)
    diff = (lateral_inp.diff_p *0.5 + diff* delt) / (delt + 0.5) 
    lateral_inp.diff_p = diff
    lateral_inp.t_p = t
    return -2000*diff
    
def lateral_inp_diff(t):
    return 0.

def lateral_inp_diff_2(t):
    return 0.

A = symbols('A')

myMBS.add_parameter_expr('phi', rotation_inp_expr(), {A: 0.05})
#myMBS.add_parameter('phi', rotation_inp, rotation_inp_diff, rotation_inp_diff_2)

myMBS.add_parameter('theta_lateral', lateral_inp, lateral_inp_diff, lateral_inp_diff_2)
myMBS.add_rotating_marker_para('tire_carrier_M0', 'tire_carrier', 'phi', 0.,-0.2,0., 'Y')
myMBS.add_body_3d('tire','tire_carrier_M0', 1.0, I_tire , 'revolute', parameters = ['Z'])
#
####################################
## some tire forces
myMBS.add_force_special('tire_carrier', 'grav')
myMBS.add_force_special('tire', 'grav')

#myMBS.add_force_special('tire_carrier', 'spring-damper-axes', parameters = [20000., -0.9, 500.])
#myMBS.add_force('tire_carrier', 'car_body_M0', parameters = [20000., 0.8, 700.])

myMBS.add_force_spline_r('tire_carrier', 'car_body_M0', DATA_PATH+"/force_spring.dat", [0.8, 1.0])
myMBS.add_force_spline_v('tire_carrier', 'car_body_M0', DATA_PATH+"/force_damper.dat", [1.0])

myMBS.add_one_body_force_model('tiremodel', 'tire', 'tire_carrier_M0', 'tire')


myMBS.add_parameter_torque('tire', 'tire_carrier_M0', [0.,0.,1.], 'theta_lateral')
##################################
# create control signals:
vel_1 = myMBS.get_body('tire').get_vel_magnitude()
myMBS.add_control_signal(vel_1)
##################################
# try to add into model
m1 = myMBS.get_model('tiremodel')
m1.add_signal(vel_1)
m1.add_signal(vel_1)
#x0 = np.hstack(( 0. * np.ones(myMBS.dof), 0. * np.ones(myMBS.dof)))
x0 = np.array([ -3.06217550e-02,  -1.60406433e-08,   8.27725196e-01,
        -3.21284978e-10,  -6.54751512e-01,  -1.59008129e+01,
        -8.35888371e-04,  -8.20890122e-10,   3.61938302e-05,
        -2.51438728e-11,  -3.30521499e-05,  -2.50105938e-09])
         

 
#for b in myMBS.bodies.keys():
#    myMBS.add_damping(b,0.1)

#################################################
# external force definitions
#def ext_sinus(t):
#    omega = 0.5*t
#    return 1000.0*np.sin(omega * t)
#    
#myMBS.add_force_ext('tire', 'world_M0', 0.,1.,0., ext_sinus)

moving_frames_in_graphics = ['tire_carrier_M0']
fixed_frames_in_graphics = []#myMBS.mames[999][0], myMBS.marker_frames[999][1]]
forces_in_graphics = ['car_body']
bodies_in_graphics = {'tire': 'tire'}
##################################################################
# constant parameters
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)

myMBS.kaneify()
##################################################################
# now the equations are fully provided with parameters in ...
#################################################

myMBS.prep_lambdas(moving_frames_in_graphics, fixed_frames_in_graphics, forces_in_graphics, bodies_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 30.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e-0)

x_final = myMBS.x_t[-1]
################################################
# linear analysis of the last state (returns also the jacobian)
jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)


myMBS.prepare(DATA_PATH, save=True)
#myMBS.plotting(t_max, dt, plots='tire')
x = myMBS.animate(t_max, dt, scale = 4, time_scale = 1.0, labels = True, t_ani = 30., center = 2)


