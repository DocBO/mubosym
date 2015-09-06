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
myMBS = mbs.MBSworld('reflective_wall', connect=False, force_db_setup=False)
#myMBS1 = MBS()

body = ["A", "B", "C"]
marker = ["A_M0", "B_M0", "C_M0"]
x0 = []

I = [1.,1.,1.]

######################################
# reflective wall reloaded
#setup clean line

#b_n[0] = myMBS.add_body_3d(999, 0, 1.0, I, 'rod-1-cardanic', parameters = [1.5,0.])
myMBS.add_body_3d(body[0], 'world_M0', 1.0, I, 'y-axes', parameters = [0.])
myMBS.add_force_special(body[0], 'grav')

myMBS.add_force(body[0],'world_M0', parameters = [100.,0.,0.])
myMBS.add_marker(marker[0], body[0], 0.,0.,0.)

for ii in range(len(body))[1:]:
    myMBS.add_body_3d(body[ii], marker[ii-1], 1.0, I, 'rod-1-cardanic', parameters = [-1.5,0.])
    myMBS.add_marker(marker[ii], body[ii],  0.,0.,0.)
    myMBS.add_force_special(body[ii], 'grav')
    


#myMBS.add_reflective_wall(b_n[1], eqn, IF, 20. )
#myMBS.add_reflective_wall(b_n[2], eqn, IF, 0. )


x0 = np.hstack(( 1. * np.ones(myMBS.dof), 0. * np.ones(myMBS.dof)))
#x0[2] = 0.
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
#str_m_b, equ, str_m_b_ref, c, gamma, s):
#eqn = x-2.0
#myMBS.add_reflective_wall(b_n[1], eqn, IF, 1000, .2 , mySyms, -1.)

#equ1 = y
#myMBS.add_geometric_constaint(2, equ1, IF, factor, mySyms)
#equ2 = x-1.5
#myMBS.add_geometric_constaint(2, equ2, IF, factor, mySyms)

#another constraint
#equ2 = x
#myMBS.add_geometric_constaint(2, equ2, IF, factor, mySyms)

#################################################
# constants
g = symbols('g')
constants = [ g ]          # Parameter definitions 
constants_vals = [9.81]     # Numerical value

const_dict = dict(zip(constants, constants_vals))  
myMBS.set_const_dict(const_dict)


#for ii in range(len(myMBS.eq_constr)/2):
#    x0 = myMBS.correct_the_initial_state(myMBS.n_constr[ii], x0)



body_frames_in_graphics = [body[0], body[1], body[2]]
fixed_frames_in_graphics = []

myMBS.kaneify()
myMBS.prep_lambdas(body_frames_in_graphics, fixed_frames_in_graphics)


dt = 0.01  # 10 ms for a nice animation result
t_max = 30.

myMBS.inte_grate_full(x0, t_max, dt, mode = 0)

x_final = myMBS.x_t[-1]


x00 = dict(zip(myMBS.q_flat+myMBS.u_flat, x_final))


x_op = {myMBS.q_flat[0]: 0.0, myMBS.q_flat[1]: np.pi/2., myMBS.q_flat[2]: np.pi/2., myMBS.u_flat[1]: 0.0, myMBS.u_flat[0]: 0.0, myMBS.u_flat[2]: 0.0}
#x_op = {myMBS.q_flat[0]: 0.0, myMBS.q_flat[1]: 0.0, myMBS.q_flat[2]: 0.0, myMBS.u_flat[1]: 0.0, myMBS.u_flat[0]: 0.0, myMBS.u_flat[2]: 0.0}

a_op = {myMBS.a_flat[0]: 0.0, myMBS.a_flat[1]: 0., myMBS.a_flat[2]: 0.}


jac = myMBS.calc_lin_analysis_n(len(myMBS.x_t)-1)



myMBS.prepare(mbs.DATA_PATH, save=True)
myMBS.animate(t_max, dt, scale = 4, time_scale = 1, t_ani = 30.0)
