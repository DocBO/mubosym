# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:43:00 2015

@author: oliver
"""

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
from interp1d_interface import interp
mbs.BASE_PATH = BASE_PATH

################################################
# test 1
################################################
print("\n\n-------------------------------")
print("TEST 1...")
# general system setup example
myMBS = mbs.MBSworld('crank_slider', connect=False, force_db_setup=False)
#prepare a standard
I = [0.,0.,0.]
######################################
# some loop constraint
myMBS.add_marker('world_M1','world', 0.,0.,0., 0., 0.,0.)
myMBS.add_body_3d('b1','world_M1', 1.0, I, 'rod-1-cardanic-efficient', parameters = [1.0,0.])
myMBS.add_marker('b1_M0','b1', 0.,0.,0.)
myMBS.add_body_3d('b2', 'b1_M0', 1.0, I, 'rod-1-cardanic-efficient', parameters = [3.0,0.])
myMBS.add_force_special('b2', 'grav')

######################################
# start conditions
x0 = np.array([-14.75044927,   1.3777362 ,   5.4077404 ,   1.50199767])

######################################
# loop constraint
factor = 20.
R = myMBS.get_frame('world_M0')
x = R[0]
y = R[1]
z = R[2]
equ1 = y
myMBS.add_geometric_constaint('b2', equ1, 'world_M0', factor)


#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions
constants_vals = [9.81]     # Numerical value
const_dict = dict(zip(constants, constants_vals))
myMBS.set_const_dict(const_dict)
myMBS.kaneify()
fixed_frames_in_graphics = ['world_M1']
frames_in_graphics = ['b1', 'b2']
forces_in_graphics = ['b1', 'b2']
myMBS.prep_lambdas(frames_in_graphics, fixed_frames_in_graphics, forces_in_graphics)
dt = 0.01  # 10 ms for a nice animation result
t_max = 15.
myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e-0)
assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([ 57.28569865,   1.82019692,   4.32724853,  -1.00045109]),myMBS.x_t[-1]))
################################################
# test 2
################################################
print("\n\n-------------------------------")
print("TEST 2...")
myMBS = mbs.MBSworld('swing_table', connect=False, force_db_setup=False)

#prepare a standard
I = [0.,0.,0.]
######################################
# some loop constraint
myMBS.add_marker('world_M1','world', 1.,0.,0.)
myMBS.add_body_3d('b1','world_M0', 1.0, I, 'rod-1-cardanic', parameters = [1.0,0.])
myMBS.add_force_special('b1', 'grav')
#
myMBS.add_marker('b1_M0','b1', 0.,0.,0.)
myMBS.add_body_3d('b2', 'b1_M0', 1.0, I, 'rod-1-cardanic', parameters = [1.5,0.])
myMBS.add_force_special('b2', 'grav')

myMBS.add_marker('b2_M0','b2', 0.,0.,0.)
myMBS.add_body_3d('b3', 'b2_M0', 1.0, I, 'rod-1-cardanic', parameters = [1.0,0.])
myMBS.add_force_special('b3', 'grav')
x0 = np.array([-13.73437514,  15.30494211,  -5.87985813,   0.81904538,
        -0.81998223,   0.82150335])
frames_in_graphics = ['b1', 'b2']

factor = 5.

R = myMBS.get_frame('world_M0')
x = R[0]
y = R[1]
z = R[2]

for b in myMBS.bodies.keys():
    myMBS.add_damping(b,0.1)
equ1 = y
myMBS.add_geometric_constaint('b3', equ1, 'world_M0', factor)
equ2 = x-1.5
myMBS.add_geometric_constaint('b3', equ2, 'world_M0', factor)
#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions
constants_vals = [9.81]     # Numerical value
const_dict = dict(zip(constants, constants_vals))
myMBS.set_const_dict(const_dict)

myMBS.kaneify()
dt = 0.01  # 10 ms for a nice animation result
t_max = 20.
myMBS.inte_grate_full(x0, t_max, dt, mode = 0)
assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([-12.27885932,  13.84937381,  -4.42472966,  -0.9882765 ,
         0.9881176 ,  -0.98782944]),myMBS.x_t[-1]))

################################################
# test 3
################################################
print("\n\n-------------------------------")
print("TEST 3...")
# general system setup example
myMBS = mbs.MBSworld('reflective_wall', connect=False, force_db_setup=False)
body = ["A", "B", "C"]
marker = ["A_M0", "B_M0", "C_M0"]
x0 = []
I = [1.,1.,1.]

######################################
# reflective wall reloaded
myMBS.add_body_3d(body[0], 'world_M0', 1.0, I, 'y-axes', parameters = [0.])
myMBS.add_force_special(body[0], 'grav')

myMBS.add_force(body[0],'world_M0', parameters = [100.,0.,0.])
myMBS.add_marker(marker[0], body[0], 0.,0.,0.)

for ii in range(len(body))[1:]:
    myMBS.add_body_3d(body[ii], marker[ii-1], 1.0, I, 'rod-1-cardanic', parameters = [1.5,0.])
    myMBS.add_marker(marker[ii], body[ii],  0.,0.,0.)
    myMBS.add_force_special(body[ii], 'grav')

x0 = np.hstack(( 1. * np.ones(myMBS.dof), 0. * np.ones(myMBS.dof)))
factor = 5.
R = myMBS.get_frame('world_M0')
x = R[0]
y = R[1]
z = R[2]
for b in myMBS.bodies.keys():
    myMBS.add_damping(b,0.1)

eqn = x-0.5
myMBS.add_reflective_wall(body[2], eqn, 'world_M0', 1000, .2, +1.)
myMBS.add_reflective_wall(body[1], eqn, 'world_M0', 1000, .2, +1.)
#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))
myMBS.set_const_dict(const_dict)

body_frames_in_graphics = [body[0], body[1], body[2]]
fixed_frames_in_graphics = []
myMBS.kaneify()
myMBS.prep_lambdas(body_frames_in_graphics, fixed_frames_in_graphics)
dt = 0.01  # 10 ms for a nice animation result
t_max = 30.
myMBS.inte_grate_full(x0, t_max, dt, mode = 0)

assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([-0.02253284,  0.48113244,  0.0802598 ,  1.54017102, -0.78752434, 1.10845035]),myMBS.x_t[-1]))
################################################
# test 4
################################################
print("\n\n-------------------------------")
print("TEST 4...")
# general system setup example
myMBS = mbs.MBSworld('strange_pendulum', connect=False, force_db_setup=False)

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
myMBS.add_body_3d(b_n[0], 'world_M1', 1.0, I , 'angle-rod', parameters = [np.pi/4., 2.0])
myMBS.add_force_special(b_n[0], 'grav')

myMBS.add_marker(m_n[0], b_n[0], 0.,0.,0.)
myMBS.add_body_3d(b_n[1], m_n[0], 1.0, I, 'angle-rod', parameters = [np.pi/4., 2.0])
myMBS.add_force_special(b_n[1], 'grav')

myMBS.add_marker(m_n[1], b_n[1], 0.,0.,0.)
myMBS.add_body_3d(b_n[2], m_n[1], 1.0, I, 'angle-rod', parameters = [np.pi/4., 2.0])
myMBS.add_force_special(b_n[2], 'grav')

x0 = np.hstack(( 1. * np.ones(myMBS.dof), 1. * np.ones(myMBS.dof)))
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

assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([ 39.37414311,   1.31856406,   0.19013064,   1.03270165,
         0.63633591,   1.45670909]),myMBS.x_t[-1]))
################################################
# test 5
################################################
print("\n\n-------------------------------")
print("TEST 5...")
# general system setup example
myMBS = mbs.MBSworld('rotating_pendulum', connect=False, force_db_setup=False)
I = [1.,1.,1.]

#############################################################
# rotating frame constraint
omega = 1.0
A = 5.5
def rotation_inp(t):
    return A*t #np.sin(omega*t)

def rotation_inp_diff(t):
    return A #*omega*np.cos(omega*t)

def rotation_inp_diff_2(t):
    return 0.# A*omega*omega*np.sin(omega*t)

myMBS.add_parameter('phi', rotation_inp, rotation_inp_diff, rotation_inp_diff_2)
myMBS.add_rotating_marker_para('rot_M0', 'world', 'phi', 0., 1.5, 0., 'Y')
myMBS.add_body_3d('mount', 'rot_M0', 1.0, I , 'rod-zero', parameters = [1.0,'X']) #[np.pi/2., 2.0])
myMBS.add_force_special('mount', 'grav')
myMBS.add_marker('mount_M0', 'mount', 0.,0.,0.)
myMBS.add_body_3d('pendulum', 'mount_M0', 1.0, I , 'rod-1-cardanic', parameters = [1.5,0.]) #[np.pi/2., 2.0])
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

assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([ 1.44551396,  0.02772908]),myMBS.x_t[-1]))
################################################
# test 6
################################################
print("\n\n-------------------------------")
print("TEST 6...")
# general system setup example
myMBS = mbs.MBSworld('planetary_char', connect=False, force_db_setup=False)
#prepare a standard
I = [1.,1.,1.]

myMBS.add_body_3d('sun', 'world_M0', 1000.0, I , 'xz-plane', parameters = [], graphics = False) #[np.pi/2., 2.0])
myMBS.add_marker('sun_M1', 'sun', 0.,0.,0.)
myMBS.add_force_special('sun', 'grav')
myMBS.add_body_3d('planet1','sun_M1', 10.0, I , 'xz-plane', parameters = [], graphics = False) #[np.pi/2., 2.0])
myMBS.add_body_3d('planet2','sun_M1', 10.0, I , 'xz-plane', parameters = [], graphics = False)
myMBS.add_force_spline_r('sun','planet1', DATA_PATH+'/force_kl1.dat', [0., -1.0])
myMBS.add_force_spline_r('sun','planet2', DATA_PATH+'/force_kl1.dat', [0., -1.0])

x0 = np.hstack(( 0.,0.,1.,1.,-1.,-1., 0.,0.,1.,0.,0.,1.))
body_frames_in_graphics = ['sun','planet1','planet2']
fixed_frames_in_graphics = []

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

assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([  4.00697011e-01,   3.95631978e-01,   5.50716034e-01,
        -1.39225303e+00,  -1.42181116e+00,   1.03779127e+00,
         1.71264875e-02,   4.34909997e-04,  -8.34620715e-01,
         3.04450091e-01,   8.77189905e-02,   6.51189090e-01]),myMBS.x_t[-1]))
################################################
# test 7
################################################
print("\n\n-------------------------------")
print("TEST 7...")
myMBS = mbs.MBSworld('quarter_car', connect=False, force_db_setup=False)
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

k = interp(filename = DATA_PATH+'/vel_01.dat')
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
myMBS.add_parameter('theta_lateral', lateral_inp, lateral_inp_diff, lateral_inp_diff_2)
myMBS.add_rotating_marker_para('tire_carrier_M0', 'tire_carrier', 'phi', 0.,-0.2,0., 'Y')
myMBS.add_body_3d('tire','tire_carrier_M0', 1.0, I_tire , 'revolute', parameters = ['Z'])
#
####################################
## some tire forces
myMBS.add_force_special('tire_carrier', 'grav')
myMBS.add_force_special('tire', 'grav')

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
myMBS.prep_lambdas(moving_frames_in_graphics, fixed_frames_in_graphics, forces_in_graphics, bodies_in_graphics)
dt = 0.01  # 10 ms for a nice animation result
t_max = 30.
myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e-0)

assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([ -5.26057927e-02,   3.51625112e+02,   7.27281000e-01,
         1.52753420e+01,  -5.54307527e-01,  -1.08426628e+03,
        -7.51480035e-04,   1.44813584e+01,   1.15400720e-05,
         1.48052321e+00,  -1.05020070e-05,  -4.41122101e+01]),myMBS.x_t[-1]))
################################################
# test 8
################################################
print("\n\n-------------------------------")
print("TEST 8...")
################################################
# test 9
################################################
print("\n\n-------------------------------")
print("TEST 9...")
myMBS = mbs.MBSworld('moving_pendulum', connect=False, force_db_setup=False)
I = [0.,0.,0.]
############################################################
# rotating frame constraint
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
myMBS.inte_grate_full(x0, t_max, dt, mode = 0)

assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([-24.93463153,   3.86271006]),myMBS.x_t[-1]))
################################################
# test 10
################################################
print("\n\n-------------------------------")
print("TEST 10...")
myMBS = mbs.MBSworld('bending_stiffness', connect=False, force_db_setup=False)
#prepare a standard
I = [10.,10.,10.]
###################################
# a complex torsional and bending stiffness example
myMBS.add_marker('world_M1', 'world', 0.,0.,1.)
myMBS.add_body_3d('rod_1', 'world_M1', 1.0, I , 'rod-1-cardanic', parameters = [0.,0.]) #[np.pi/2., 2.0])
myMBS.add_torque_3d('rod_1', 'bending-stiffness-1', parameters=[np.pi,800.])# [0.,0.,0.]])
myMBS.add_force_special('rod_1', 'grav')
myMBS.add_marker('rod_1_M0', 'rod_1', 0.,0.,0.)
myMBS.add_body_3d('rod_2', 'rod_1_M0', 1.0, I, 'rod-1-cardanic', parameters = [1.0,np.pi/2.])
myMBS.add_torque_3d('rod_2', 'bending-stiffness-1', parameters=[0.,800.])
myMBS.add_force_special('rod_2', 'grav')
myMBS.add_marker('rod_2_M0', 'rod_2', 0.,0.,0.)
myMBS.add_body_3d('rod_3', 'rod_2_M0', 1.0, I, 'rod-zero', parameters = [2.0, 'Z'])
myMBS.add_force_special('rod_3', 'grav')
myMBS.add_marker('rod_3_M0', 'rod_3', 0.,0.,0.)
myMBS.add_body_3d('rod_4', 'rod_3_M0', 1.0, I, 'rod-1-revolute', parameters = [0.5,0.,2.0])
myMBS.add_force_special('rod_4', 'grav')
myMBS.add_torque_3d('rod_4', 'rotation-stiffness-1', parameters = [100.])

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
myMBS.kaneify()

moving_frames_in_graphics = ['rod_1','rod_2','rod_3','rod_4']
fixed_frames_in_graphics = []
myMBS.prep_lambdas(moving_frames_in_graphics, fixed_frames_in_graphics)
dt = 0.01  # 10 ms for a nice animation result
t_max = 30.
myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e-0)

assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([  3.19277659e+00,  -7.33029363e-05,   3.60462376e-01,
        -8.19189775e-06,   4.91640339e-04,   1.62021365e+00]),myMBS.x_t[-1]))
################################################
# test 11
################################################
print("\n\n-------------------------------")
print("TEST 11...")
myMBS = mbs.MBSworld('kreisel', connect=False, force_db_setup=False)
#prepare a standard
I = [50.,50.,50.]
I0 = [1.,1.,1.]
###################################
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
myMBS.kaneify()
moving_frames_in_graphics = ['B3']
fixed_frames_in_graphics = []
myMBS.prep_lambdas(moving_frames_in_graphics, fixed_frames_in_graphics)
dt = 0.01  # 10 ms for a nice animation result
t_max = 15.
myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e+0)
x_final = myMBS.x_t[-1]
assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([  7.99061700e-04,   5.88066100e-01,  -1.94878723e-03,
         1.50000000e+03,   1.74841210e-04,   7.83931952e-02,
        -2.78589420e-02,   1.00000000e+02]),myMBS.x_t[-1]))
################################################
# test 12
################################################
# general system setup example
print("\n\n-------------------------------")
print("TEST 12...")
myMBS = mbs.MBSworld('chain_6', connect=False, force_db_setup=False)
b_n = []
m_n = []
b_n_max = 6
for ii in range(b_n_max):
    b_n.append(str(ii))
    m_n.append(str(ii)+"_M0")
#prepare a standard
I = [0.,0.,0.]
######################################
# large chain
myMBS.add_marker('world_M1','world', 0.,0.,0., 0., 0.,0.)
myMBS.add_body_3d(b_n[0],'world_M1', 1.0, I, 'rod-1-cardanic-efficient', parameters = [1.0,0.])
myMBS.add_force_special(b_n[0], 'grav')
for ii in range(0,b_n_max-1):
    myMBS.add_marker(m_n[ii],b_n[ii], 0.,0.,0.)
    myMBS.add_body_3d(b_n[ii+1], m_n[ii], 1.0, I, 'rod-1-cardanic-efficient', parameters = [1.0,0.])
    myMBS.add_force_special(b_n[ii+1], 'grav')
x0 = np.hstack(( 0. * np.ones(myMBS.dof), 4. * np.ones(myMBS.dof)))
for b in myMBS.bodies.keys():
    myMBS.add_damping(b,0.4)
#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions
constants_vals = [9.81]     # Numerical value
const_dict = dict(zip(constants, constants_vals))
myMBS.set_const_dict(const_dict)
fixed_frames_in_graphics = ['world_M1']
frames_in_graphics = []
forces_in_graphics = []
myMBS.kaneify()
myMBS.prep_lambdas(frames_in_graphics, fixed_frames_in_graphics, forces_in_graphics)
dt = 0.01  # 10 ms for a nice animation result
t_max = 20.
myMBS.inte_grate_full(x0, t_max, dt, mode = 0, tolerance = 1e+0)

assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([-0.0104617 ,  6.26186732,  6.32782105,  6.22693558,  6.28696372,
        6.32940285,  0.06457042, -0.14155971,  0.5408486 , -0.24090285,
       -0.07871267,  0.19237471]),myMBS.x_t[-1]))

################################################
# test 13
################################################
print("\n\n-------------------------------")
print("TEST 13...")
myMBS = mbs.MBSworld('new_constraint', connect=False, force_db_setup=False)
#prepare a standard
I = [0.,0.,0.]
######################################
# choose
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
assert(t_max == myMBS.time[-1])
assert(np.allclose(np.array([ 0.28300594, -0.95923776, -4.16528328, -1.22898956]),myMBS.x_t[-1]))

print("\n\n All tests ok !!! you are allowed to do your pull request ... but only in develop branch")
