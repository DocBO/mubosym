# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:41:09 2015

@author: oliver
"""

import numpy as np
from sympy import symbols, sin

import mubosym as mbs
from interp1d_interface import interp
from driving_line import interp_dl
###############################################################
# general system setup example
myMBS = mbs.MBSworld('simple_car_sequ', connect=True, force_db_setup=False)


axes_rear_marker = ['body_car_rr', 'body_car_rl' ]
axes_front_marker = ['body_car_fr', 'body_car_fl' ]

I_car = [500.,3000.,1500.]
I_0 = [0.,0.,0.] 
I_tire = [1.,1.,1.]

k = interp(filename = mbs.DATA_PATH+"/vel_01.dat")
#high end definition of static variables...
@mbs.static_vars(t_p=0, diff_p=0)
def lateral_inp(t): 
    #return -20.
    velocity = myMBS.get_control_signal(0)
    v_soll = k.f_interp(t)
    diff = (v_soll-velocity)
    delt = (t-lateral_inp.t_p)
    diff = (lateral_inp.diff_p *0.5 + diff* delt) / (delt + 0.5) 
    lateral_inp.diff_p = diff
    lateral_inp.t_p = t
    return -400*diff
 
def zero(t):
    return 0.
    
myMBS.add_parameter('theta_lateral', lateral_inp, zero, zero)
c_0, h, gamma = symbols('c_0, h, gamma')

def axes_front(axes_marker, n):
    myMBS.add_body_3d('carrier_f'+str(n), axes_marker, 60.0, I_0, 'y-axes', parameters = [])
    myMBS.add_force_special('carrier_f'+str(n), 'spring-damper-axes', parameters = [c_0, -h, gamma])
    myMBS.add_force_special('carrier_f'+str(n), 'grav')
    myMBS.add_rotating_marker_para('carrier_M_f'+str(n), 'carrier_f'+str(n), 'phi', 0.,-0.20,0.0, 'Y')
    myMBS.add_body_3d('tire_f'+str(n), 'carrier_M_f'+str(n), 1.0, I_tire , 'revolute-Z', parameters = [])
    myMBS.add_one_body_force_model('tiremodel_'+str(n), 'tire_f'+str(n),'carrier_M_f'+str(n),'tire')
    
def axes_rear(axes_marker, n):
    myMBS.add_body_3d('carrier_r'+str(n), axes_marker, 60.0, I_0, 'y-axes', parameters = [])
    myMBS.add_force_special('carrier_r'+str(n), 'spring-damper-axes', parameters = [c_0, -h, gamma])
    myMBS.add_force_special('carrier_r'+str(n), 'grav')
    myMBS.add_marker('carrier_M_r'+str(n), 'carrier_r'+str(n) , 0.,-0.20,0.0)
    myMBS.add_body_3d('tire_r'+str(n), 'carrier_M_r'+str(n), 1.0, I_tire , 'revolute-Z', parameters = [])
    myMBS.add_one_body_force_model('tiremodel_'+str(n), 'tire_r'+str(n),'carrier_M_r'+str(n),'tire')
    
# a simple car using sequence buildup
myMBS.add_body_3d('body_car', 'world_M0', 1700.0, I_car, 'free-6', parameters = [], graphics = False) #[np.pi/2., 2.0])
myMBS.add_marker('body_car_fr', 'body_car', 1.5,0.,0.7)
myMBS.add_marker('body_car_fl', 'body_car', 1.5,0.,-0.7)
myMBS.add_marker('body_car_rr', 'body_car', -1.5,0.,0.7)
myMBS.add_marker('body_car_rl', 'body_car', -1.5,0.,-0.7)

myMBS.add_force_special('body_car', 'grav')

###############################################
##steering expression:
#def rotation_inp_expr():
#    t, A = symbols('t A')
#    return A*sin(1.0*t)
#A = symbols('A')
#def rotation_inp_expr():
#    t, A = symbols('t A')
#    return (A+0.02*t)*sin(1.0*t)
    
#A = -0.02
#omega = 0.8
#def rotation_inp(t):
#    if t < 10.:
#        return 0.
#    else:
#        return A*np.sin(omega*(t-10.))
#    
#def rotation_inp_diff(t):
#    if t < 10.:
#        return 0.
#    else:
#        return A*omega*np.cos(omega*(t-10.))
#
#def rotation_inp_diff_2(t):
#    if t < 10.:
#        return 0.
#    else:
#        return -A*omega*omega*np.sin(omega*(t-10.))

###################################
# steering controller
name = mbs.DATA_PATH+'/line_01.dat'
kst = interp_dl(filename = name)

@mbs.static_vars(t_p=0, dist_p=0, tau=0, phi_out=0., phi_int=0.)
def rotation_inp(t):
    bz = myMBS.get_control_signal(1)
    bx = myMBS.get_control_signal(2)
    vz = myMBS.get_control_signal(3)
    vx = myMBS.get_control_signal(4)
    delt = (t-rotation_inp.t_p)
    if delt > 0.05: #0.1
        rotation_inp.tau, x, z, dist = kst.distance((bx,bz), rotation_inp.tau)
        t_diff = kst.tang_diff((vx,vz), rotation_inp.tau)
        #dist = (rotation_inp.dist_p *0.1 + dist* delt) / (delt + 0.1) 
        rotation_inp.t_p = t
        #rotation_inp.dist_p = dist
        rotation_inp.phi_int -= dist/1e+3
        rotation_inp.phi_out = t_diff/0.5e+2 + rotation_inp.phi_int - dist/1e+3
        if rotation_inp.phi_out > 0.12:
            rotation_inp.phi_out = 0.12
        elif rotation_inp.phi_out < -0.12:
            rotation_inp.phi_out = -0.12
            
        print t, delt, rotation_inp.phi_out, dist, t_diff
    return rotation_inp.phi_out
    #return A*np.sin(omega*t)

def zero(t):
    return 0.

#myMBS.add_parameter_expr('phi', rotation_inp_expr(), {A: 0.0})
myMBS.add_parameter('phi', rotation_inp, zero, zero)
n = 0
for name in axes_rear_marker:
    axes_rear(name, n)
    n+=1
n = 0
for name in axes_front_marker:
    axes_front(name, n)
    n+=1

myMBS.add_parameter_torque('tire_r0', 'carrier_M_r0', [0.,0.,1.], 'theta_lateral')
myMBS.add_parameter_torque('tire_r1', 'carrier_M_r1', [0.,0.,1.], 'theta_lateral')

#################################################
# constants
g = symbols('g')
height = 0.4
constants = [ g,  c_0, h, gamma ]             # Constant definitions 
constants_vals = [9.81, 35000, height, 4500 ]      # Numerical values

const_dict = dict(zip(constants, constants_vals))
myMBS.set_const_dict(const_dict)

##################################
# create control signals:
vel_1 = myMBS.get_body('body_car').get_vel_magnitude()
myMBS.add_control_signal(vel_1, "Geschw.", "m/s")
dist_z = myMBS.get_body('body_car').z()
myMBS.add_control_signal(dist_z, "Abstand zur Spur", "m")
dist_x = myMBS.get_body('body_car').x()
myMBS.add_control_signal(dist_x, "Fahrstrecke", "m")
vz = myMBS.get_body('body_car').z_dt()
vx = myMBS.get_body('body_car').x_dt()
myMBS.add_control_signal(vz, "vz", "m/s")
myMBS.add_control_signal(vx, "vx", "m/s")

##################################


#to settle car ...
for b in myMBS.bodies.keys():
    myMBS.add_damping(b, 10.0)
    
body = myMBS.get_body('body_car')
#body.set_small_angles([body.get_phi(), body.get_psi()])

myMBS.kaneify(simplify=False)

moving_frames_in_graphics = ['tire_f0', 'carrier_M_f0', 'body_car']
fixed_frames_in_graphics = []
forces_in_graphics = ['body_car']
bodies_in_graphics = {'tire_f0': 'tire','tire_f1': 'tire','tire_r0': 'tire','tire_r1': 'tire', 'body_car':'box'}

myMBS.prep_lambdas(moving_frames_in_graphics, fixed_frames_in_graphics, forces_in_graphics, bodies_in_graphics)
############################################################
# now setup all maneuvers

#myMBS.exchange_parameter_expr('phi', rotation_inp_expr(), {A: 0.05})
dt = 0.01  # 10 ms for a nice animation result
t_max = 25.
#first starting conditions ... (thrown on the street)
#x0 = np.hstack(( 0. * np.ones(myMBS.dof), 1. * np.zeros(myMBS.dof)))



x0 = np.array([ -6.98702128e-08,   5.85260703e-06,  -3.62073872e-07,
         1.12804336e-04,   height+0.3,   9.40668424e-08,
        -height,  -3.32459591e-04,  -height,
        -3.31175253e-04,  -height,  -3.33075785e-04,
        -height,  -3.31802373e-04,  -2.89428308e-07,
        -6.98425700e-08,  -2.59191657e-06,   5.04886037e-06,
         2.39749721e-07,  -2.31595563e-07,  -3.82747611e-06,
        -5.06125891e-06,  -3.57698842e-06,  -5.35752999e-06,
         3.04894626e-06,  -5.07140660e-06,   3.26773587e-06,
        -5.36760469e-06])

myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e+1)

x_final = myMBS.x_t[-1]
################################################
# linear analysis of the last state (returns also the jacobian)
jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)

inp = raw_input("Weiter zur Animation (return)...")

myMBS.prepare(mbs.DATA_PATH, save=False)
myMBS.plotting(t_max, dt, plots='signals')
#a = myMBS.animate(t_max, dt, scale = 4, time_scale = 1, t_ani = 35., labels = True, center = 0)
