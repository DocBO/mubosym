# -*- coding: utf-8 -*-
"""
all mubosym related core classes
================================
Created on Sun Mar  8 11:50:46 2015

@author: oliver
"""
from __future__ import print_function, absolute_import
import os.path,sys,time,copy

sys.path.insert(0,os.path.realpath(os.path.dirname(__file__)))

import b_splines_interface #as kennlinie
import one_body_force_model_interface # as one_body_force_model
import simple_tire_model_interface # simple_tire_model

try:
    import pandas as pd
    no_pandas = False
except:
    print( 'can not use pandas here' )
    no_pandas = True

from sympy import symbols, lambdify, sign, re, acos, asin, sin, cos, Poly
from sympy.physics.mechanics import ( Vector, ReferenceFrame, Point, dynamicsymbols, outer,
                        RigidBody, KanesMethod, gradient)
from sympy.solvers import solve as sp_solve

#Vector.simp = True

from numpy import array, hstack, vstack, ones, zeros, linspace, pi, sqrt
from numpy.linalg import eig
from numpy.linalg import solve as np_solve
from scipy.linalg import solve as sc_solve #, lu_solve
#from scipy.sparse.linalg import factorized
from scipy.integrate import ode, odeint

import mubosym as mbs

#import matplotlib as mp
#mp.use('WX')
from matplotlib import pyplot as plt
#########################################
from dbconnect import dbHandler
from symTools import list_to_world, worldData
#########################################
from interp1d_interface import interp
from one_body_force_model_interface import one_body_force_model
from simple_tire_model_interface import simple_tire_model

### Imports always on top...
from vpython_3d import animation

class ParameterError(Exception):
    """
    Exception raised for errors in the parameters

    :param paras: input expression in which the error occurred
    :param n_soll: explanation of the error
    """

    def __init__(self, paras, n_soll, name):
        self.msg = name+"... caused by: "+str(paras)+". Please give me "+str(n_soll)+" parameters"
        super(ParameterError, self).__init__(self.msg)

class InputError(Exception):
    """
    Exception raised for errors in the parameters

    :param paras: input expression in which the error occurred
    :param n_soll: explanation of the error
    """
    def __init__(self, input):
        self.msg = 'Wrong Input caused by: ' + input
        super(InputError, self).__init__(self.msg)

IF = ReferenceFrame('IF')              # Inertial reference frame
O = Point('O')                         # Origin point
O.set_vel(IF, 0)                       # Origin's velocity is zero
g, t = symbols('g t')                  # Gravity and time

class MBSframe(ReferenceFrame):
    """
    This class represents a moving frame. (sympy only provides rotating frames up to now)
    """
    def __init__(self,Name):
        global IF, O, g, t
        ReferenceFrame.__init__(self,Name)
        self.Orig = Point('O_'+Name)
        self.Orig.set_pos(O, 0.*IF.x)
        self.phi = 0.
        self.pos = 0.*IF.x
        self.vel = 0.*IF.x
        self.acc = 0.*IF.x
    def set_pos_vec_IF(self, vec):
        self.pos = vec
        self.vel = vec.diff(t, IF)
        self.acc = self.vel.diff(t, IF)
        self.Orig.set_pos(O, vec)
        self.Orig.set_vel(IF, self.vel)
    def set_vel_vec_IF(self, vec):
        self.Orig.set_vel(IF, vec)
        self.vel = vec
    def get_vel_vec_IF(self):
        return self.Orig.vel(IF)
    def set_pos_Pt(self, Pt):
        self.Orig = Pt
    def get_pos_Pt(self):
        return self.Orig
    def get_pos_IF(self):
        return self.Orig.pos_from(O).express(IF)
    def get_pos_SELF(self):
        return self.Orig.pos_from(O).express(self)
    def express_vec_in(self, vec):
        return vec.express(self, variables=True) - self.get_pos_SELF()
    def get_omega(self, frame):
        return self.ang_vel_in(frame)
    def get_ex_IF(self):
        return self.x.express(IF)
    def get_ey_IF(self):
        return self.y.express(IF)
    def get_ez_IF(self):
        return self.z.express(IF)
    def px(self):
        return self.pos.dot(IF.x)
    def py(self):
        return self.pos.dot(IF.y)
    def pz(self):
        return self.pos.dot(IF.z)
    def px_dt(self):
        return self.vel.dot(IF.x)
    def py_dt(self):
        return self.vel.dot(IF.y)
    def pz_dt(self):
        return self.vel.dot(IF.z)
    def px_ddt(self):
        return self.acc.dot(IF.x)
    def py_ddt(self):
        return self.acc.dot(IF.y)
    def pz_ddt(self):
        return self.acc.dot(IF.z)
    def set_dicts(self, dicts):
        for d in dicts:
            self.pos = self.pos.subs(d)
            self.vel = self.vel.subs(d)
            self.acc = self.acc.subs(d)

class MBSbody(object):
    def __init__(self, n, name, mass, I, pos, vel, frame, joint, N_att, N_att_fixed, atm = ''):
        self.n = n
        self.frame = frame
        self.joint = joint
        self.vel = vel
        self.pos = pos
        self.name = name
        self.mass = mass
        self.I = I
        self.N_att = N_att
        self.N_att_fixed = N_att_fixed
        self.attached_to_marker = atm #string name of marker
    def get_vel(self):
        return self.vel
    def get_vel_magnitude(self):
        return self.vel.magnitude()
    def get_pos(self):
        return self.pos
    def Pt(self):
        return self.frame.get_pos_Pt()
    def x(self):
        return self.frame.px()
    def y(self):
        return self.frame.py()
    def z(self):
        return self.frame.pz()
    def x_dt(self):
        return self.frame.px_dt()
    def y_dt(self):
        return self.frame.py_dt()
    def z_dt(self):
        return self.frame.pz_dt()
    def x_ddt(self):
        return self.frame.px_ddt()
    def y_ddt(self):
        return self.frame.py_ddt()
    def z_ddt(self):
        return self.frame.pz_ddt()
    def get_frame(self):
        return self.frame
    def get_n(self):
        return self.n
    def get_N_att(self):
        return self.N_att
    def get_N_att_fixed(self):
        return self.N_att_fixed
    def get_mass(self):
        return self.mass
    def set_dicts(self, dicts):
        for d in dicts:
            self.pos = self.pos.subs(d)
            self.vel = self.vel.subs(d)

class MBSmarker(object):
    def __init__(self, name, frame, body_name):
        self.name = name
        self.frame = frame
        self.body_name = body_name   #here the name is better
    def get_frame(self):
        return self.frame
    def get_body_name(self):
        return self.body_name
    def Pt(self):
        return self.frame.get_pos_Pt()
    def x(self):
        return self.frame.px()
    def y(self):
        return self.frame.py()
    def z(self):
        return self.frame.pz()
    def x_dt(self):
        return self.frame.px_dt()
    def y_dt(self):
        return self.frame.py_dt()
    def z_dt(self):
        return self.frame.pz_dt()
    def x_ddt(self):
        return self.frame.px_ddt()
    def y_ddt(self):
        return self.frame.py_ddt()
    def z_ddt(self):
        return self.frame.pz_ddt()
    #def set_dicts(self, dicts):
    #    for d in dicts:
    #        self.pos = self.pos.subs(d)
    #        self.vel = self.vel.subs(d)

class MBSparameter(object):
    def __init__(self, name, sym, sym_dt, sym_ddt, func, func_dt, func_ddt, diff_dict, const = 0.):
        self.name = name
        self.sym = sym
        self.sym_dt = sym_dt
        self.sym_ddt = sym_ddt
        self.func = func
        self.func_dt = func_dt
        self.func_ddt = func_ddt
        self.diff_dict = diff_dict
        self.c = const
    def get_func(self):
        return self.func, self.func_dt, self.func_ddt
    def get_paras(self):
        return self.sym, self.sym_dt, self.sym_ddt
    def get_diff_dict(self):
        return self.diff_dict
    def set_constant(self, c):
        self.c = c

class MBSmodel(object):
    def __init__(self, name, reference):
        self.name = name
        self.ref = reference
    def add_signal(self, expr):
        self.ref.add_signal(expr)

class MBSio(object):
    """
    This class creates a dummy MBSworld object and returns it for the user to play with.

    But maybee an IO handling class should be created instead?

    imagine just passing a MBSworld object and executing animate() from here
    """
    def __init__(self,filename,MBworld=None,save=False,params = ['state', 'orient', 'con', 'e_kin', 'time', 'x_t', 'acc', 'e_pot', 'e_tot', 'e_rot', 'speed', 'signals']):

        if MBworld == None:
            pass
            #self.MBworld = object()
        if not save:
            self.__read__(filename,params)
        if save:
            self.store = pd.HDFStore(filename,complevel=9, complib='bzip2', fletcher32=True)
            self.__save__(params,MBworld)

    def __save__(self,params,MBworld):
        """
        creates a Store object or buffer
        """
#        for par in params:
#            self.store[par] = getattr(MBworld,par)
#
#        self.store.close()

        self.store['state'] = pd.DataFrame(MBworld.state[:,:3],columns=['x', 'y', 'z']) # 3 includes 3d cartesians
        self.store['orient'] = pd.DataFrame(MBworld.orient[:,:6],columns=['ex_x', 'ex_y', 'ex_z', 'eys_x', 'ey_y', 'ey_z']) #2 cartesians vectors e_x, e_y
        self.store['con']  = pd.DataFrame(MBworld.con)  # 3d cartesian vector from-to (info)

        self.store['e_kin'] = pd.DataFrame(MBworld.e_kin)
        self.store['time'] = pd.DataFrame(MBworld.time)
        self.store['x_t'] = pd.DataFrame(MBworld.x_t)
        self.store['acc'] = pd.DataFrame(MBworld.acc)
        self.store['e_pot'] = pd.DataFrame(MBworld.e_pot)
        self.store['e_tot'] = pd.DataFrame(MBworld.e_tot)
        self.store['e_rot'] = pd.DataFrame(MBworld.e_rot)
        self.store['speed'] = pd.DataFrame(MBworld.speed)
        self.store['signals'] = pd.Series(MBworld.signals)

#        here we must consider on how to store the data properly...
#        self.store['vis_body_frames'] = pd.DataFrame(self.vis_body_frames) #1 frame moving
#        self.store['vis_forces'] = pd.DataFrame(self.vis_forces) # 1 force on body
#        self.store['vis_frame_coords'] = pd.DataFrame(self.vis_frame_coords)
#        self.store['vis_force_coords'] = pd.DataFrame(self.vis_force_coords)


    def __read__(self,filename,params):
        for par in params:
            setattr(self,par,pd.read_hdf(filename,par))
        #return self
            #here we must consider on how to store the data properly...
            #self.store['vis_body_frames'] = pd.DataFrame(self.vis_body_frames) #1 frame moving
            #self.store['vis_forces'] = pd.DataFrame(self.vis_forces) # 1 force on body
            #self.store['vis_frame_coords'] = pd.DataFrame(self.vis_frame_coords)
            #self.store['vis_force_coords'] = pd.DataFrame(self.vis_force_coords)

    def animate(self):
        pass

class MBSworld(object):
    """
    All storages lists and dictionaries for a full multibody sim world. Keeps track of every frame, body, force and torque.
    Includes also parameters and external model interface (e.g. tire).
    """
    def __init__(self, name = '', connect=False, force_db_setup=False):
# setup the world
        global IF, O, g, t
        self.n_body = -1                            # number of the actual body
        self.name = name
        self.mydb = None
        self.connect = connect
        if connect:
            self.mydb = dbHandler(mbs.BASE_PATH+'/data')
            if not self.mydb.has_key(name) or force_db_setup:
                self.db_setup = True
            else:
                self.db_setup = False
        self.bodies = {}  # name mapping
        self.parameters = []       #keep the order correct, initialized in kaneify
        self.parameters_diff = {}  #helper dict, initialized in kaneify
        self.q = []       # generalized coordinates nested list
        self.u = []       # generalized velocities nested list
        self.a = []       # generalized accelerations nested list
        self.q_flat = []  # generalized coordinates flat list
        self.u_flat = []  # generalized velocities flat list
        self.a_flat = []  # generalized accelerations flat list

        self.f_ext_act = []       # the actually added external forces (needed scalar symbols)
        self.f_int_act = []       # the actually added internal forces (needed scalar symbols)
        self.m = []     # Mass of each body
        self.Ixx = []   # Inertia xx for each body
        self.Iyy = []   # Inertia yy for each body
        self.Izz = []   # Inertia zz for each body
        self.forces_ext_n = 0 #number of actual external forces (counting)
        self.forces_int_n = 0 #number of actual internal forces (counting)
        self.forces_models_n = 0   #number of model input functions
        self.f_int_expr = []  #to keep expression of all dofs results in one scalar var
        self.f_int_lamb = []  #same but lambidified, input always the full state vec (even if not all of it is used)
        self.f_int_func = []  #storage for the function handles (for the one scalar corresponding in f_int_lamb)
        self.f_ext_func = []  #to keep some func references (python function handles)
        self.f_t_models_sym = [] #general storage list for symbols for model forces
        self.models = []        #general storage list for model handlers
        self.models_obj = {}     #storage dict for model objects
        self.f_models_lamb = []
        self.kl = None
        self.forces = []
        self.torques = []
        self.particles = []
        self.kindiffs = []
        self.kindiff_dict = {}
        self.accdiff = []
        self.accdiff_dict = {}
        self.body_frames = {}
        self.body_list_sorted = []
        self.bodies_in_graphics = {}
        self.eq_constr = []
        self.n_constr = []
        self.dof = 0
        self.IF_mbs = MBSframe('IF_mbs')
        self.IF_mbs.orient(IF, 'Axis', [0.,IF.z])
        self.IF_mbs.set_pos_Pt(O)

        self.pot_energy_saver = []
        self.con_type = [] #just for grafics
        self.body_frames[999] = self.IF_mbs


        self.bodies.update({'world':999})

        self.tau_check = 0.
        self.control_signals = []         # here the numbers are stored
        self.control_signals_lamb = []    # here the lambdas for signals are stored
        self.control_signals_expr = []    # here the expressions are created/stored
        # new obj-paradigm starts here
        self.bodies_obj = {}
        self.marker_obj = {}
        self.marker_fixed_obj = {}
        self.param_obj = [] #name, pos, vel, frame, joint, ref_frame):
        self.bodies_obj.update({'world': MBSbody(999,'world', 0., [0.,0.,0.], self.IF_mbs.get_pos_IF(),self.IF_mbs.get_vel_vec_IF(),self.IF_mbs,'',self.IF_mbs,self.IF_mbs)})
        self.add_marker('world_M0', 'world',0.,0.,0.)

    def add_control_signal(self, expr):
        self.control_signals.append(0.)
        self.control_signals_expr.append(expr)

    def add_parameter(self, new_para_name, fct_para, fct_para_dt , fct_para_ddt):
        global IF, O, g, t
        params_n = len(self.param_obj)
        p = dynamicsymbols('para_'+str(params_n))
        p_dt = dynamicsymbols('para_dt'+str(params_n))
        p_ddt = dynamicsymbols('para_ddt'+str(params_n))
        diff_dict = {p.diff(): p_dt, p_dt.diff(): p_ddt}
        self.param_obj.append(MBSparameter(new_para_name,p,p_dt,p_ddt,fct_para,fct_para_dt,fct_para_ddt, diff_dict))

    def add_parameter_expr(self, new_para_name, expression, const = {}):
        global IF, O, g, t
        params_n = len(self.param_obj)
        p = dynamicsymbols('para_'+str(params_n))
        p_dt = dynamicsymbols('para_dt'+str(params_n))
        p_ddt = dynamicsymbols('para_ddt'+str(params_n))
        diff_dict = {p.diff(): p_dt, p_dt.diff(): p_ddt}
        expression = expression.subs(const)
        dt0 = lambdify(t,expression)
        dt1 = lambdify(t,expression.diff(t))
        dt2 = lambdify(t,expression.diff(t).diff(t))
        self.param_obj.append(MBSparameter(new_para_name,p,p_dt,p_ddt,dt0,dt1,dt2,diff_dict))

    def exchange_parameter(self, para_name, fct_para, fct_para_dt , fct_para_ddt):
        pobj = [ o for o in self.param_obj if o.name == para_name ][0]
        pobj.func = dt0
        pobj.func_dt = dt1
        pobj.func_ddt = dt2

    def exchange_parameter_expr(self, para_name, expression, const = {}):
        global IF, O, g, t
        expression = expression.subs(const)
        dt0 = lambdify(t,expression)
        dt1 = lambdify(t,expression.diff(t))
        dt2 = lambdify(t,expression.diff(t).diff(t))
        pobj = [ o for o in self.param_obj if o.name == para_name ][0]
        pobj.func = dt0
        pobj.func_dt = dt1
        pobj.func_ddt = dt2

    def add_rotating_marker_para(self, new_marker_name, str_n_related, para_name, vx, vy, vz, axes):
        """
        Add a rotating marker framer with parameter related phi

        :param str new_marker_name: the new marker name
        :param str str_n_related: the reference fixed body frame
        :param str para_name: the parameter name which is the rotation angle
        :param float vx, vy, vz: the const. translation vector components in the related frame
        :param str axes:
        """
        try:
            body = self.bodies_obj[str_n_related]
        except:
            raise InputError("no such body (make body first) %s" % str_n_related)
        try:
            phi = [x.sym for x in self.param_obj if x.name == para_name][0]
        except:
            raise InputError("no such parameter name %s" % str(para_name))
        MF = MBSframe('MF_'+new_marker_name)
        MF_fixed = MBSframe('MF_fixed_'+new_marker_name)
        N_fixed = body.get_frame()
        n_related = body.get_n()
        if axes == 'Y':
            MF.orient(N_fixed,'Body',[0.,phi,0.], 'XYZ')
            MF_fixed.orient(N_fixed,'Body',[0.,phi,0.], 'XYZ')
        elif axes == 'Z':
            MF.orient(N_fixed,'Body',[0.,0.,phi], 'XYZ')
            MF_fixed.orient(N_fixed,'Body',[0.,0.,phi], 'XYZ')
        elif axes == 'X':
            MF.orient(N_fixed,'Body',[phi, 0.,0.], 'XYZ')
            MF_fixed.orient(N_fixed,'Body',[phi, 0.,0.], 'XYZ')
        pos_0 = N_fixed.get_pos_IF()
        pos = pos_0 + vx * N_fixed.x + vy * N_fixed.y + vz * N_fixed.z
        vel = pos.diff(t, IF)
        MF.set_pos_vec_IF(pos)
        #MF.set_vel_vec_IF(vel)
        MF_fixed.set_pos_vec_IF(pos)
        #MF_fixed.set_vel_vec_IF(vel)
        self.marker_obj.update({new_marker_name:MBSmarker(new_marker_name, MF, str_n_related)})
        self.marker_fixed_obj.update({new_marker_name:MBSmarker(new_marker_name, MF_fixed, str_n_related)})


    def add_moving_marker(self, new_marker_name, str_n_related, vx, vy, vz, vel, acc, axes):
        """
        Add a moving marker framer via velocity and acceleration.

        :param str new_marker_name: the name of the new marker
        :param str str_n_related: the reference fixed body frame
        :param float vx, vy, vz: is the const. translation vector in the related frame
        :param float vel: the velocity of the marker
        :param float acc: the acceleration of the marker
        :param str axes: the axis where the velocity is oriented ('X', 'Y' or 'Z')
        """
        global IF, O, g, t
        try:
            body = self.bodies_obj[str_n_related]
        except:
            raise InputError("no such body (make body first) %s" % str_n_related)

        MF = MBSframe('MF_'+new_marker_name)
        MF_fixed = MBSframe('MF_fixed_'+new_marker_name)
        N_fixed = body.get_frame()
        MF.orient(N_fixed,'Body',[0.,0.,0.], 'XYZ')
        MF_fixed.orient(N_fixed,'Body',[0.,0.,0.], 'XYZ')
        (kx,ky,kz) = dynamicsymbols('kx, ky, kz')
        if axes == 'X':
            kx = vx + vel * t + 0.5* acc * t*t
            ky = vy
            kz = vz
        elif axes == 'Y':
            kx = vx
            ky = vy + vel * t + 0.5* acc * t*t
            kz = vz
        elif axes == 'Z':
            kx = vx
            ky = vy
            kz = vz + vel * t + 0.5* acc * t*t
        pos_0 = N_fixed.get_pos_IF()
        pos = pos_0 + kx * N_fixed.x + ky * N_fixed.y + kz * N_fixed.z
        vel = pos.diff(t, IF)
        MF.set_pos_vec_IF(pos)
        #MF.set_vel_vec_IF(vel)
        MF_fixed.set_pos_vec_IF(pos)
        #MF_fixed.set_vel_vec_IF(vel)
        self.marker_obj.update({new_marker_name:MBSmarker(new_marker_name, MF, str_n_related)})
        self.marker_fixed_obj.update({new_marker_name:MBSmarker(new_marker_name, MF_fixed, str_n_related)})

    def add_moving_marker_para(self, new_marker_name, str_n_related, para_name, vx, vy, vz, axes):
        """
        Add a moving marker framer via a parameter.

        :param str new_marker_name: the name of the new marker
        :param str str_n_related: the reference fixed body frame
        :param str para_name: the name of the involved parameter (as position)
        :param float vx, vy, vz: the const. translation vector components in the related frame
        :param str axes: the axis where the parameter as velocity is oriented ('X', 'Y' or 'Z')
        """
        global IF, O, g, t
        try:
            body = self.bodies_obj[str_n_related]
        except:
            raise InputError("no such body (make body first) %s" % str_n_related)
        try:
            phi = [x.sym for x in self.param_obj if x.name == para_name][0]
        except:
            raise InputError("no such parameter name %s" % str(para_name))
        MF = MBSframe('MF_'+new_marker_name)
        MF_fixed = MBSframe('MF_fixed_'+new_marker_name)
        N_fixed = body.get_frame()
        MF.orient(N_fixed,'Body',[0.,0.,0.], 'XYZ')
        MF_fixed.orient(N_fixed,'Body',[0.,0.,0.], 'XYZ')
        (kx,ky,kz) = dynamicsymbols('kx, ky, kz')
        if axes == 'X':
            kx = vx + phi
            ky = vy
            kz = vz
        elif axes == 'Y':
            kx = vx
            ky = vy + phi
            kz = vz
        elif axes == 'Z':
            kx = vx
            ky = vy
            kz = vz + phi
        else:
            raise InputError("please use 'X', 'Y' or 'Z' for axes")
        pos_0 = N_fixed.get_pos_IF()
        pos = pos_0 + kx * N_fixed.x + ky * N_fixed.y + kz * N_fixed.z
        vel = pos.diff(t, IF)
        MF.set_pos_vec_IF(pos)
        #MF.set_vel_vec_IF(vel)
        MF_fixed.set_pos_vec_IF(pos)
        #MF_fixed.set_vel_vec_IF(vel)
        self.marker_obj.update({new_marker_name:MBSmarker(new_marker_name, MF, str_n_related)})
        self.marker_fixed_obj.update({new_marker_name:MBSmarker(new_marker_name, MF_fixed, str_n_related)})

    def add_rotating_marker(self, new_marker_name, str_n_related, vx, vy, vz, omega, axes):
        """
        Add a rotating marker framer with constant omega.

        :param str new_marker_name: the name of the new marker
        :param str str_n_related: the reference fixed body frame
        :param float vx, vy, vz: the const. translation vector components in the related frame (float*3)
        :param float omega: the angular velocity
        :param str axes: the axis where the angular velocity is oriented ('X', 'Y' or 'Z')
        """
        global IF, O, g, t
        try:
            body = self.bodies_obj[str_n_related]
        except:
            raise InputError("no such body (make body first) %s" % str_n_related)

        MF = MBSframe('MF_'+new_marker_name)
        MF_fixed = MBSframe('MF_fixed_'+new_marker_name)
        N_fixed = body.get_frame()
        if axes == 'Y':
            MF.orient(N_fixed,'Body',[0.,omega*t,0.], 'XYZ')
            MF_fixed.orient(N_fixed,'Body',[0.,omega*t,0.], 'XYZ')
        elif axes == 'Z':
            MF.orient(N_fixed,'Body',[0.,0.,omega*t], 'XYZ')
            MF_fixed.orient(N_fixed,'Body',[0.,0.,omega*t], 'XYZ')
        elif axes == 'X':
            MF.orient(N_fixed,'Body',[omega*t, 0.,0.], 'XYZ')
            MF_fixed.orient(N_fixed,'Body',[omega*t, 0.,0.], 'XYZ')
        else:
            raise InputError("as axis enter one of these: X, Y, or Z")

        pos_0 = N_fixed.get_pos_IF()
        pos = pos_0 + vx * N_fixed.x + vy * N_fixed.y + vz * N_fixed.z
        vel = pos.diff(t, IF)
        MF.set_pos_vec_IF(pos)
        #MF.set_vel_vec_IF(vel)
        MF_fixed.set_pos_vec_IF(pos)
        #MF_fixed.set_vel_vec_IF(vel)
        self.marker_obj.update({new_marker_name:MBSmarker(new_marker_name, MF, str_n_related)})
        self.marker_fixed_obj.update({new_marker_name:MBSmarker(new_marker_name, MF_fixed, str_n_related)})

    def add_marker(self, new_marker_name, str_n_related, vx, vy, vz, phix = 0., phiy = 0., phiz = 0.):
        """
        Add a fixed marker framer related to a body frame. Marker frames are used to add joints or forces (they usually act between two of them or a body and a marker).

        :param str new_marker_name: the name of the new marker
        :param str str_n_related: the reference fixed body frame
        :param float vx, vy, vz: the const. translation vector components in the related frame (float*3)
        :param float phix,phiy,phiz: the const. rotation Eulerian angles for a new (body-fixed) orientation of the marker frame
        """
        global IF, O, g, t
        try:
            body = self.bodies_obj[str_n_related]
        except:
            raise InputError("no such body (make body first) %s" % str_n_related)
        MF = MBSframe('MF_'+new_marker_name)
        MF_fixed = MBSframe('MF_fixed_'+new_marker_name)
        N_fixed = body.get_frame()
        MF.orient(N_fixed,'Body',[phix,phiy,phiz], 'XYZ')
        MF_fixed.orient(N_fixed,'Body',[phix,phiy,phiz], 'XYZ')
        pos_0 = N_fixed.get_pos_IF()
        pos = pos_0 + vx * N_fixed.x + vy * N_fixed.y + vz * N_fixed.z
        vel = pos.diff(t, IF)
        MF.set_pos_vec_IF(pos)
        #MF.set_vel_vec_IF(vel)
        MF_fixed.set_pos_vec_IF(pos)
        #MF_fixed.set_vel_vec_IF(vel)
        self.marker_obj.update({new_marker_name:MBSmarker(new_marker_name, MF, str_n_related)})
        self.marker_fixed_obj.update({new_marker_name:MBSmarker(new_marker_name, MF_fixed, str_n_related)})

    def _interpretation_of_str_m_b(self,str_m_b):
        """
        Relates the name string of a marker or body to an internal number and frame and a boolean which indicates the type (body or marker)

        :param str_m_b: the name string of marker or body
        """
        obj = None
        if self.bodies_obj.has_key(str_m_b):
            try:
                obj = body = self.bodies_obj[str_m_b]
                n = body.get_n()             #self.bodies[str_m_b]
                N_fixed_n = body.get_frame() #self.body_frames[n]
                is_body = True
            except:
                raise InputError("body frame not existent for name %s" % str_m_b)
        else:
            try:
                obj = marker = self.marker_fixed_obj[str_m_b]
                b_name = marker.get_body_name()
                n = self.bodies_obj[b_name].get_n()
                N_fixed_n = marker.get_frame()
                is_body = False
            except:
                raise InputError("marker frame not existent for name %s" % str_m_b)

        return n, N_fixed_n, is_body, obj

    def add_body_3d(self, new_body_name, str_n_marker, mass, I , joint, parameters = [], graphics = True):
        """
        Core function to add a body for your mbs model. Express the pos and vel of the body in terms of q and u (here comes the joint crunching).
        Generalized coordinates q and u are often (not always) written in the rotated center of mass frame of the previous body.

        :param str new_body_name: the name of the new body (freely given by the user)
        :param str str_n_marker: the name string of the marker where the new body is related to (fixed on)
        :param float mass: the mass of the new body
        :param float*3 I: the inertia of the new body measured in the body symmetric center of mass frame (float*3 list) [Ixx, Iyy, Izz]
        :param str joint: the type of the joint: possible choices are (see code and examples). You can add whatever joint you want to. Here the degrees of freedom are generated.
        :param float*n parameters: the parameters to descripe the joint fully (see code)
        :param str graphics: the choice of the grafics representative (ball, car, tire)
        """
        global IF, O, g, t
        if not str_n_marker in self.marker_obj:
            raise InputError("marker frame not existent for name %s" % str_n_marker)
        #print n_marker, n_att
        if graphics:
            self.con_type.append(joint)
        else:
            self.con_type.append('transparent')
        free0 = [ 'rod-zero' ]
        free1 = [ 'rod-1-cardanic-efficient', 'axes', 'y-axes', 'x-axes', 'z-axes', 'angle-rod', 'revolute', 'rod-1-revolute', 'rod-1-cardanic']
        free2 = [ 'rod-2-cardanic', 'rod-2-revolute-scharnier' , 'xy-plane', 'xz-plane', 'yx-plane', 'free-2-rotate']
        free3 = [ 'spring-rod', 'free-3-rotate', 'rod-3-revolute-revolute', 'free-3-translate' ]
        free4 = [ 'free-3-translate-z-rotate' ]
        free6 = [ 'free-6' ]
        self.n_body += 1
        n_body = self.n_body  # number of the actual body (starts at 0)
        # create correct number of symbols for the next body
        if joint in free0:
            d_free = 0
        elif joint in free1:
            d_free = 1
        elif joint in free2:
            d_free = 2
        elif joint in free3:
            d_free = 3
        elif joint in free4:
            d_free = 4
        elif joint in free6:
            d_free = 6
        else:
            raise InputError(joint)
        self.q.append(dynamicsymbols('q'+str(n_body)+'x0:'+str(d_free)))
        self.u.append(dynamicsymbols('u'+str(n_body)+'x0:'+str(d_free)))
        self.a.append(dynamicsymbols('a'+str(n_body)+'x0:'+str(d_free)))
        self.dof += d_free
        self.m.append(mass)
        self.Ixx.append(I[0])
        self.Iyy.append(I[1])
        self.Izz.append(I[2])

        # add the center of mass point to the list of points
        Pt = Point('O_'+new_body_name)
        # we need previous cm_point to find origin of new frame
        N_att = self.marker_obj[str_n_marker].get_frame()
        vec_att = N_att.get_pos_IF()
        N_att_fixed = self.marker_fixed_obj[str_n_marker].get_frame()
        #create frame for each body on the center of mass and body fixed
        N_fixed = MBSframe('N_fixed_'+new_body_name)
        N_fixed.set_pos_Pt(Pt)
        self.body_frames[n_body] = N_fixed
        t_frame = vec_att

        #express the velocity and position in terms of the marker frame
        # not to forget the body fixed frame
        if joint == 'x-axes':
            t_frame = parameters[0]*IF.y
            N_att.orient(IF, 'Axis', [0., IF.z] )
            pos_pt = (self.q[n_body][0]*N_att.x).express(IF, variables = True)+t_frame
        elif joint == 'angle-rod': # parameter[0] theta, parameter[1] length
            N_fixed.orient(N_att_fixed, 'Body', [self.q[n_body][0],-parameters[0],0.], 'YXZ' )
            N_att.orient(N_att_fixed, 'Body', [self.q[n_body][0],-parameters[0],0.], 'YXZ' )
            pos_pt = (parameters[1]*N_att.y).express(IF, variables = True)+t_frame
        elif joint == 'y-axes':
            N_fixed.orient(N_att_fixed, 'Body', [0.,0.,0.], 'YXZ' )
            N_att.orient(N_att_fixed, 'Body', [0.,0.,0.], 'YXZ' )
            pos_pt = (self.q[n_body][0]*N_att.y).express(IF, variables = True)+t_frame
        elif joint == 'xz-plane':
            N_fixed.orient(IF, 'Axis', [0.,IF.z] )
            N_att.orient(IF, 'Axis', [0., IF.z] )
            pos_pt = (self.q[n_body][0]*N_att.x+self.q[n_body][1]*N_att.z).express(IF, variables = True)+t_frame
        elif joint == 'xy-plane':
            N_fixed.orient(IF, 'Axis', [0.,IF.z] )
            N_att.orient(IF, 'Axis', [0., IF.z] )
            pos_pt = (self.q[n_body][0]*N_att.x+self.q[n_body][1]*N_att.y).express(IF, variables = True)+t_frame
        elif joint == 'free-3-translate-z-rotate':
            N_fixed.orient(N_att_fixed, 'Body', [self.q[n_body][0],0.,0.], 'YXZ' )
            N_att.orient(N_att_fixed, 'Body', [self.q[n_body][0],0.,0.], 'YXZ' )
            pos_pt = (self.q[n_body][1]*IF.x + self.q[n_body][2]*IF.y + self.q[n_body][3]*IF.z).express(IF, variables = True)+t_frame
        elif joint == 'free-3-translate':
            N_fixed.orient(N_att_fixed, 'Axis', [0., IF.y] )
            N_att.orient(N_att_fixed, 'Axis', [0., IF.y] ) #phi, r
            pos_pt = (self.q[n_body][0]*IF.x + self.q[n_body][1]*IF.y + self.q[n_body][2]*IF.z).express(IF, variables = True)+t_frame

        elif joint == 'free-6':
            N_fixed.orient(IF, 'Body', [self.q[n_body][0],self.q[n_body][1],self.q[n_body][2]], 'XYZ' )
            pos_pt = (self.q[n_body][3]*IF.x + self.q[n_body][4]*IF.y + self.q[n_body][5]*IF.z).express(IF, variables = True)+t_frame
        elif joint == 'rod-1-cardanic-efficient':
            N_fixed.orient(IF, 'Axis', [self.q[n_body][0], IF.z] )
            N_att.orient(IF, 'Axis', [self.q[n_body][0], IF.z] ) #phi, r
            #N_att.set_ang_vel(IF, self.u[n_body][0] * IF.z)
            pos_pt = (-parameters[0]*N_att.y).express(IF, variables = True)+t_frame
        #note there are 2 types of technical spherical movement: cardanic and revolute-scharnier
        elif joint == 'rod-1-cardanic': # parameter[0]: length, parameters[1]: phi
            N_fixed.orient(N_att_fixed, 'Body', [parameters[1],self.q[n_body][0],0.], 'YZX' )
            N_att.orient(N_att_fixed, 'Body', [parameters[1],self.q[n_body][0],0.], 'YZX' )
            pos_pt = (-parameters[0]*N_att.y).express(IF, variables = True)+t_frame
        elif joint == 'rod-2-cardanic': # parameter[0]: length
            N_fixed.orient(N_att_fixed, 'Body', [self.q[n_body][0],self.q[n_body][1],0.], 'ZXY' )
            N_att.orient(N_att_fixed, 'Body', [self.q[n_body][0],self.q[n_body][1],0.], 'ZXY' )
            pos_pt = (-parameters[0]*N_att.y).express(IF, variables = True)+t_frame
        elif joint == 'rod-2-revolute-scharnier': # parameter[0]: length
            N_fixed.orient(N_att_fixed, 'Body', [self.q[n_body][0],self.q[n_body][1],0.], 'YXZ' )
            N_att.orient(N_att_fixed, 'Body', [self.q[n_body][0],self.q[n_body][1],0.], 'YXZ' )
            pos_pt = (-parameters[0]*N_att.y).express(IF, variables = True)+t_frame
        elif joint == 'free-3-rotate': # now working (not for bending stiffness)
            N_fixed.orient(IF, 'Body', [self.q[n_body][0],self.q[n_body][1],self.q[n_body][2]], 'XYZ' )
            pos_pt = t_frame
        elif joint == 'rod-1-revolute':# parameter[0]: length
            N_fixed.orient(N_att_fixed, 'Body', [self.q[n_body][0],0.,0.], 'YXZ' )
            N_att.orient(N_att_fixed, 'Body', [self.q[n_body][0],0.,0.], 'YXZ' )
            pos_pt = (-parameters[0]*N_att.y).express(IF, variables = True)+t_frame
        elif joint == 'revolute': # parameter[0]: 'X' or 'Y' or 'Z'
            if parameters[0] == 'X':
                orient = 'XYZ'
            elif parameters[0] == 'Y':
                orient = 'YXZ'
            elif parameters[0] == 'Z':
                orient = 'ZXY'
            N_fixed.orient(N_att_fixed, 'Body', [self.q[n_body][0],0.,0.], orient )
            N_att.orient(N_att_fixed, 'Body', [self.q[n_body][0],0.,0.], orient )
            pos_pt = t_frame
        elif joint == 'rod-zero':# parameter[0]: length, parameter[1]: direction
            N_fixed.orient(N_att_fixed, 'Body', [0.,0.,0.], 'XYZ')
            if parameters[1] == 'X':
                pos_pt = (parameters[0]*N_fixed.x).express(IF, variables = True)+t_frame
            elif parameters[1] == 'Y':
                pos_pt = (parameters[0]*N_fixed.y).express(IF, variables = True)+t_frame
            else:
                pos_pt = (parameters[0]*N_fixed.z).express(IF, variables = True)+t_frame

        vel_pt = pos_pt.diff(t, IF)
        self.bodies_obj.update({new_body_name:MBSbody(n_body,new_body_name, mass, I, pos_pt, vel_pt, N_fixed, joint, N_att, N_att_fixed, str_n_marker) })
        print( "body no: ",n_body )
        print( "body pos: ", pos_pt )
        print( "body vel: ", vel_pt )

        #TODO check the coefficients and delete all small ones

        #Pt.set_pos(O, pos_pt)              # Set the position of Pt
        #Pt.set_vel(IF, vel_pt)             # Set the velocity of Pt
        N_fixed.set_pos_vec_IF(pos_pt)

        Ixx = self.Ixx[n_body]*outer(N_fixed.x, N_fixed.x)
        Iyy = self.Iyy[n_body]*outer(N_fixed.y, N_fixed.y)
        Izz = self.Izz[n_body]*outer(N_fixed.z, N_fixed.z)
        I_full = Ixx + Iyy + Izz
        Pa = RigidBody('Bd' + str(n_body), Pt, N_fixed, self.m[n_body], (I_full, Pt))

        self.particles.append(Pa)
        for ii in range(d_free):
            self.kindiffs.append(self.q[n_body][ii].diff(t) - self.u[n_body][ii])
            self.accdiff.append(self.u[n_body][ii].diff(t) - self.a[n_body][ii])
        self.bodies.update({new_body_name:n_body})
        return n_body

    def get_body(self, name):
        return self.bodies_obj[name]

    def get_model(self, name):
        return self.models_obj[name]

    def add_force(self, str_m_b_i, str_m_b_j, parameters = []):
        """
        Interaction forces between body/marker i and j via spring damper element.

        :param str str_m_b_i: name of the body/marker i
        :param str str_m_b_j: name of the body/marker j
        :param list parameters: stiffness, offset, damping-coefficient
        """
        eps = 1e-2
        if not len(parameters) == 3:
            raise ParameterError(parameters, 2, "add_force")
        i, N_fixed_i, _, body_i = self._interpretation_of_str_m_b(str_m_b_i)
        j, N_fixed_j, _, body_j = self._interpretation_of_str_m_b(str_m_b_j)

        Pt_i = N_fixed_i.get_pos_Pt()
        Pt_j = N_fixed_j.get_pos_Pt()

        r_ij = Pt_i.pos_from(Pt_j)
        abs_r_ij = r_ij.magnitude()
        vel = N_fixed_i.get_vel_vec_IF() - N_fixed_j.get_vel_vec_IF()
        force = -parameters[0]*(r_ij-parameters[1]*r_ij.normalize()) - parameters[2]*vel
        self.pot_energy_saver.append(0.5*parameters[0]*(abs_r_ij-parameters[1])**2)
        self.forces.append((Pt_i,force))
        self.forces.append((Pt_j,-force))

    def add_force_special(self, str_m_b, force_type, parameters = []):
        """
        Specialised forces on body (for shortcut input)

        :param str str_m_b: is the name string of the body
        :param str force_type: the type of force which is added
        :param list parameters: corresponding parameters to describe the force
        """
        global IF, O, g, t
        n, N_fixed_n, is_body, body = self._interpretation_of_str_m_b(str_m_b)
        if not is_body:
            raise InputError("only a body name here (no marker)")

        Pt_n = body.Pt()
        N_att = body.get_N_att()
        Pt_att = N_att.get_pos_Pt()
        #print ( N_att, Pt_att )
        if force_type == 'spring-axes':
            if len(parameters) == 0 :
                raise ParameterError(parameters, 1, "spring-axes")
            force = -parameters[0]*N_fixed_n.y*(self.q[n][0]-parameters[1])
            self.pot_energy_saver.append(0.5*parameters[0]*(self.q[n][0]-parameters[1])**2)
            self.forces.append((Pt_att,-force))
        if force_type == 'spring-damper-axes':
            if len(parameters) == 0 :
                raise ParameterError(parameters, 1, "spring-damper-axes")
            force = (-parameters[0]*(self.q[n][0]-parameters[1])-parameters[2]*self.u[n][0])*N_fixed_n.y
            #print force, Pt_att, Pt_n
            self.pot_energy_saver.append(0.5*parameters[0]*(self.q[n][0]-parameters[1])**2)
            self.forces.append((Pt_att,-force))
        if force_type == 'spring-y':
            if len(parameters) == 0 :
                raise ParameterError(parameters, 1, "spring-y")
            force = -parameters[0]*IF.y*self.q[n][0]
            self.pot_energy_saver.append(0.5*parameters[0]*self.q[n][0]**2)
        if force_type == 'grav':
            force = -g*self.m[n]*IF.y
            self.pot_energy_saver.append(g*self.m[n]*self.get_pt_pos(n,IF,1))
        if force_type == 'spring-rod':
            if len(parameters) < 2:
                raise ParameterError(parameters, 2, "spring-rod")
            force = parameters[1]*(self.q[n][1]-parameters[0])*N_att.y
            self.pot_energy_saver.append(0.5*parameters[1]*(self.q[n][1]-parameters[0])**2)
            self.forces.append((Pt_att,-force))
        if force_type == 'spring-damper-rod':
            if len(parameters) < 3:
                raise ParameterError(parameters, 3, "spring-damper-rod")
            force = parameters[1]*(self.q[n][1]-parameters[0])*N_att.y+parameters[2]*self.u[n][1]*N_att.y
            self.pot_energy_saver.append(0.5*parameters[1]*(self.q[n][1]-parameters[0])**2)
            self.forces.append((Pt_att,-force))
        if force_type == 'spring-horizontal-plane':
            if len(parameters) == 0 : #parameters[0] = D, paramters[1] = y0
                raise ParameterError(parameters, 1, "spring-plane")
            y_body = N_fixed_n.get_pos_IF().dot(IF.y)-parameters[1]
            force = -parameters[0]*IF.y*y_body*(1.-re(sign(y_body))*0.5)
            self.pot_energy_saver.append(0.5*parameters[0]*y_body**2*(1.-re(sign(y_body)))*0.5)
        if force_type == 'spring-damper-horizontal-plane':
            if len(parameters) == 0 : #parameters[0] = D, paramters[1] = y0
                raise ParameterError(parameters, 1, "spring-plane")
            y_body = N_fixed_n.get_pos_IF().dot(IF.y)-parameters[1]
            v_body = y_body.diff()
            force = IF.y*(-parameters[0]*y_body-parameters[2]*v_body)*(1.-re(sign(y_body)*0.5))#+f_norm)
            self.pot_energy_saver.append(0.5*parameters[0]*y_body**2*(1.-re(sign(y_body))*0.5))

        self.forces.append((Pt_n,force))



    def add_torque_3d(self, str_m_b, torque_type, parameters = []):
        global IF, O, g, t
        n, N_fixed_n, _, body = self._interpretation_of_str_m_b(str_m_b)

        N_fixed_m = body.get_N_att_fixed()  #self.body_frames[m]


        if torque_type == 'bending-stiffness-1':
            print( 'stiffness' )
            phi = self.q[n][0]
            torque = -parameters[1]*( phi - parameters[0])*N_fixed_n.z # phi is always the first freedom
            self.pot_energy_saver.append(0.5*parameters[1]*(phi-parameters[0])**2)
            self.torques.append((N_fixed_n, torque))
            self.torques.append((N_fixed_m, -torque))

        elif torque_type == 'bending-stiffness-2':
            print( 'stiffness-2' )
            #r_vec = Pt_n.pos_from(Pt_m)

            phi = -N_fixed_n.y.cross(N_att_fixed.y)
            phi_m = asin(phi.magnitude())
            #phi = phi.normalize()
            torque = -parameters[1]*phi # phi is always the first freedom
            self.pot_energy_saver.append(0.5*parameters[1]*phi_m**2)
            self.torques.append((N_fixed_n, torque))
            self.torques.append((N_fixed_m, -torque))
            print( "TORQUE: ", phi, torque )
        elif torque_type == 'rotation-stiffness-1':
            print( 'rot-stiffness-1' )
            phi = self.q[n][0]
            torque = -parameters[0]*phi*N_fixed_m.y #-parameters[1]*phi_3

            self.pot_energy_saver.append(0.5*parameters[0]*phi**2)

            self.torques.append((N_fixed_n, torque))
            self.torques.append((N_fixed_m, -torque))
            print( "TORQUE: ", phi, "...", torque )
        else:
            raise InputError(torque_type)

    def add_parameter_torque(self, str_m_b, str_m_b_ref, v, para_name):
        """
        Add an external torque to a body in the direction v, the abs value is equal the value of the parameter (para_name)

        :param str_m_b: a body name or a marker name
        :param str_m_b_ref: a body name or a marker name as a reference where vel and omega is transmitted to the model and in which expressed the force and torque is acting
        :param v: direction of the torque (not normalized)
        :param para_name: name of the parameter
        """
        global IF, O, g, t
        try:
            phi = [x.sym for x in self.param_obj if x.name == para_name][0]
        except:
            raise InputError("no such parameter name %s" % str(para_name))
        n, N_fixed_n, _, body = self._interpretation_of_str_m_b(str_m_b)
        m, N_fixed_m, _, _ = self._interpretation_of_str_m_b(str_m_b_ref)

        n_vec = v[0] * N_fixed_m.x + v[1] * N_fixed_m.y + v[2] * N_fixed_m.z
        #print "TT1: ",n_vec*phi
        self.torques.append((N_fixed_n, n_vec*phi))

    def add_one_body_force_model(self, model_name, str_m_b, str_m_b_ref, typ='tire', parameters = []):
        """
        Add an (external) model force/torque for one body: the force/torque is acting
        on the body if the parameter is a body string and on the marker if it is a marker string

        :param str_m_b: a body name or a marker name
        :param str_m_b_ref: a body name or a marker name as a reference where vel and omega is transmitted to the model and in which expressed the force and torque is acting
        :param typ: the type of the model (the types can be extended easily, you are free to provide external models via the interface)
        :param parameters: a dict of parameters applied to the force model as initial input
        """
        global IF, O, g, t
        F = []
        T = []

        self.forces_models_n += 1
        n, N_fixed_n, _, body = self._interpretation_of_str_m_b(str_m_b)
        m, N_fixed_m, _, _ = self._interpretation_of_str_m_b(str_m_b_ref)

        F = dynamicsymbols('F_models'+str(n)+"x0:3")
        T = dynamicsymbols('T_models'+str(n)+"x0:3")
        self.f_t_models_sym = self.f_t_models_sym + F + T

        Pt_n = N_fixed_n.get_pos_Pt()
        # prepare body symbols
        pos = N_fixed_n.get_pos_IF()
        pos_x = pos.dot(IF.x)
        pos_y = pos.dot(IF.y)
        pos_z = pos.dot(IF.z)

        vel = N_fixed_n.get_vel_vec_IF()
        vel_x = vel.dot(N_fixed_m.x)
        vel_y = vel.dot(N_fixed_m.y)
        vel_z = vel.dot(N_fixed_m.z)

        omega = N_fixed_n.get_omega(IF)
        omega_x = omega.dot(N_fixed_m.x)
        omega_y = omega.dot(N_fixed_m.y)
        omega_z = omega.dot(N_fixed_m.z)

        n_vec = N_fixed_n.z
        n_x = n_vec.dot(IF.x)
        n_y = n_vec.dot(IF.y)
        n_z = n_vec.dot(IF.z)

        #get the model and supply the trafos
        if typ == 'general':
            oo = one_body_force_model()
            self.models.append(oo)
            self.models_obj.update({model_name: MBSmodel(model_name, oo) })

        elif typ == 'tire':
            oo = simple_tire_model()
            self.models.append(oo)
            self.models_obj.update({model_name: MBSmodel(model_name, oo) })

        self.models[-1].set_coordinate_trafo([pos_x, pos_y, pos_z, n_x, n_y, n_z, vel_x, vel_y, vel_z, omega_x, omega_y, omega_z])

        force = self.f_t_models_sym[-6]*N_fixed_m.x + self.f_t_models_sym[-5]*N_fixed_m.y + self.f_t_models_sym[-4]*N_fixed_m.z
        torque = self.f_t_models_sym[-3]*N_fixed_m.x + self.f_t_models_sym[-2]*N_fixed_m.y + self.f_t_models_sym[-1]*N_fixed_m.z
        #print "TTT: ",torque
        self.forces.append((Pt_n,force))
        self.torques.append((N_fixed_n, torque))

    def add_force_spline_r(self, str_m_b_i, str_m_b_j, filename, param = [0., 1.0]):
        """
        Interaction forces between body/marker i and j via characteristic_line class interp (ind. variable is the distance of the two bodies)

        :param str str_m_b_i: name of the body/marker i
        :param str str_m_b_j: name of the body/marker j
        """
        i, N_fixed_i, _, body = self._interpretation_of_str_m_b(str_m_b_i)
        j, N_fixed_j, _, _ = self._interpretation_of_str_m_b(str_m_b_j)

        Pt_i = N_fixed_i.get_pos_Pt()
        Pt_j = N_fixed_j.get_pos_Pt()

        r_ij = Pt_i.pos_from(Pt_j)
        abs_r_ij = r_ij.magnitude()
        self.f_int_act.append(dynamicsymbols('f_int'+str(i)+str(j)+'_'+str(self.forces_int_n)))
        self.forces_int_n += 1
        #one is sufficient as symbol
        force = param[1]*self.f_int_act[-1]*r_ij/(abs_r_ij+1e-3)
        kl = interp(filename)
        self.f_int_expr.append(abs_r_ij-param[0])
        self.f_int_func.append(kl.f_interp)
        #actio = reactio !!
        self.forces.append((Pt_i,force))
        self.forces.append((Pt_j,-force))

    def add_force_spline_v(self, str_m_b_i, str_m_b_j, filename, param = [1.0]):
        """
        Interaction forces between body/marker i and j via characteristic_line class interp (ind. variable is the relative velocity of the two bodies)

        :param str str_m_b_i: name of the body/marker i
        :param str str_m_b_j: name of the body/marker j
        """
        i, N_fixed_i, _, body = self._interpretation_of_str_m_b(str_m_b_i)
        j, N_fixed_j, _, _ = self._interpretation_of_str_m_b(str_m_b_j)

        Pt_i = N_fixed_i.get_pos_Pt()
        Pt_j = N_fixed_j.get_pos_Pt()

        r_ij = Pt_i.pos_from(Pt_j)
        abs_r_ij = r_ij.magnitude()
        abs_r_ij_dt = abs_r_ij.diff()
        self.f_int_act.append(dynamicsymbols('f_int'+str(i)+str(j)+'_'+str(self.forces_int_n)))
        self.forces_int_n += 1
        #one is sufficient as symbol
        force = param[0]*self.f_int_act[-1]*r_ij/abs_r_ij
        kl = interp(filename)
        self.f_int_expr.append(abs_r_ij_dt)
        self.f_int_func.append(kl.f_interp)
        #actio = reactio !!
        self.forces.append((Pt_i,force))
        self.forces.append((Pt_j,-force))

    def add_force_ext(self, str_m_b, str_m_b_ref, vx, vy, vz, py_function_handler):
        '''
        Includes the symbol f_ext_n for body/marker str_m_b expressed in frame str_m_b_ref in direction vx, vy, vz

        :param str str_m_b: name of the body/marker
        :param str str_m_b_ref: existing reference frame
        :param float vx, vy, vz: const. vector components giving the direction in the ref frame
        '''
        global IF, O, g, t
        n, N_fixed_n, _, body = self._interpretation_of_str_m_b(str_m_b)
        m, N_fixed_m, _, _ = self._interpretation_of_str_m_b(str_m_b_ref)

        #old: Pt_n = self.body_frames[n].get_pos_Pt()
        print( n, N_fixed_n )
        Pt_n = N_fixed_n.get_pos_Pt()
        self.f_ext_act.append(dynamicsymbols('f_ext'+str(n)))
        f_vec = vx * self.f_ext_act[-1]*N_fixed_m.x.express(IF) +\
                vy * self.f_ext_act[-1]*N_fixed_m.y.express(IF) +\
                vz * self.f_ext_act[-1]*N_fixed_m.z.express(IF)
        self.forces_ext_n += 1
        self.forces.append((Pt_n,f_vec))
        self.f_ext_func.append(py_function_handler)

    def get_frame(self, str_m_b):
        m, frame, _, _ = self._interpretation_of_str_m_b(str_m_b)
        return frame

    def add_geometric_constaint(self, str_m_b, equ, str_m_b_ref, factor):
        """
        Function to add a geometric constraint (plane equation in the easiest form)

        :param str str_m_b: name of the body/marker for which the constraint is valid
        :param sympy-expression equ: is of form f(x,y,z)=0,
        :param str str_m_b_ref: reference frame
        :param float factor: the factor of the constraint force (perpendicular to the plane). If this is higher, the constraint is much better fullfilled, but with much longer integration time (since the corresponding eigenvalue gets bigger).
        """
        global IF, O, g, t
        n, N_fixed_n, is_body, body = self._interpretation_of_str_m_b(str_m_b)
        m, frame_ref, _, _ = self._interpretation_of_str_m_b(str_m_b_ref)
        #frame_ref = IF
        x = frame_ref[0]
        y = frame_ref[1]
        z = frame_ref[2]
        #################################
        #valid for planes
        #nx = equ.coeff(x,1)
        #ny = equ.coeff(y,1)
        #nz = equ.coeff(z,1)
        #nv = (nx*frame_ref.x + ny*frame_ref.y + nz*frame_ref.z).normalize()
        #################################
        #valid in general
        nv = gradient(equ, frame_ref).normalize()
        self.equ = nv.dot(frame_ref.x)*x + nv.dot(frame_ref.y)*y + nv.dot(frame_ref.z)*z
        Pt = N_fixed_n.get_pos_Pt()
        vec = Pt.pos_from(O).express(frame_ref, variables = True)
        x_p = vec.dot(IF.x)
        y_p = vec.dot(IF.y)
        z_p = vec.dot(IF.z)
        d_c1 = equ.subs({x:x_p, y:y_p, z:z_p}) #.simplify()
        d_c2 = d_c1.diff(t)
        self.nv = nv = nv.subs({x:x_p, y:y_p, z:z_p})
        C = 1000. * factor
        if is_body:
            gamma = 2.*sqrt(self.m[n]*C)
        else:
            gamma = 2.*sqrt(C)
        #check for const. forces (e.g. grav) -> Projectiorator ...
        for ii in range(len(self.forces)):
            if self.forces[ii][0] == Pt:
                proj_force = - self.forces[ii][1].dot(nv)*nv
                self.forces.append((Pt, proj_force))
        #first add deviation force
        self.forces.append((Pt,(- C * d_c1  - gamma * d_c2)*factor*nv))
        #second project the inertia forces on the plane
        # Note:
        # here the number C_inf is in theory infinity to project correctly:
        # any large number can only be applied for states which are in accordance with the constraint
        # otherwise the constraint is not valid properly
        #alpha = symbols('alpha')
        #myn = -alpha*nv
        C_inf = 10.0 * factor
        proj_force = - C_inf * self.m[n]*d_c2.diff(t)*nv #- self.m[n]*acc.dot(nv)*nv
        #self.forces.append((Pt, myn))
        self.forces.append((Pt, proj_force))

        self.eq_constr.append(d_c1)
        self.n_constr.append(n)

    def add_reflective_wall(self, str_m_b, equ, str_m_b_ref, c, gamma, s):
        """
        Function to add a reflective wall constraint (plane equation in the easiest form)

        :param str str_m_b: name of the body/marker
        :param sympy-expression equ: is of form f(x,y,z)=0,
        :param str str_m_b_ref: reference frame
        :param float c: the stiffness of the reflective wall
        :param float gamma: the damping (only perpendicular) of the wall
        :param float s: the direction of the force (one of (1,-1))
        """
        global IF, O, g, t
        n, N_fixed_n, _, body = self._interpretation_of_str_m_b(str_m_b)
        m, frame_ref, _, _ = self._interpretation_of_str_m_b(str_m_b_ref)
        x = frame_ref[0]
        y = frame_ref[1]
        z = frame_ref[2]
        nx = equ.coeff(x,1)
        ny = equ.coeff(y,1)
        nz = equ.coeff(z,1)
        nv = (nx*frame_ref.x + ny*frame_ref.y + nz*frame_ref.z).normalize()

        Pt = N_fixed_n.get_pos_Pt()
        vec = Pt.pos_from(O).express(frame_ref, variables = True)
        x_p = vec.dot(IF.x)
        y_p = vec.dot(IF.y)
        z_p = vec.dot(IF.z)
        d_c1 = equ.subs({x:x_p, y:y_p, z:z_p})  #.simplify()
        d_c2 = d_c1.diff(t)
        self.forces.append((Pt,(-c * d_c1 - gamma * d_c2)*(1.-s*re(sign(d_c1)))*nv))

    def add_damping(self, str_m_b, gamma):
        """
        Add damping for the body

        :param str str_m_b: name of the body
        :param float gamma: the damping coefficient (acting in all directions)
        """
        n, N_fixed_n, is_body, body = self._interpretation_of_str_m_b(str_m_b)
        if not is_body:
            raise InputError("Damping mus be add on the body")
        Pt = body.Pt() #self.body_frames[n].get_pos_Pt()
        damp = -gamma * body.get_vel()
        self.forces.append((Pt, damp))

    def get_pt_pos(self, n, frame, coord):
        '''
        coord is 0..2
        '''
        global IF, O, g, t
        Pt = self.body_frames[n].get_pos_Pt()
        if coord == 0:
            v = frame.x
        elif coord == 1:
            v = frame.y
        elif coord == 2:
            v = frame.z
        return (Pt.pos_from(O).express(frame, variables = True).dot(v)) #.simplify()

    def get_pt_vel(self, n, frame, coord):
        '''
        coord is 0..2 (x,y,z-direction)
        '''
        s_coord = self.get_pt_pos(n, frame, coord)
        return s_coord.diff(t).subs(self.kindiff_dict)


    def get_pt_acc(self, n, frame, coord):
        v_coord = self.get_pt_vel(n, frame, coord)
        return v_coord.diff(t).subs(self.kindiff_dict).subs(self.accdiff_dict)

    def get_pt_acc_IF(self, n):
        global IF, O, g, t
        v_coord_0 = self.get_pt_vel(n, IF, 0)
        v_coord_1 = self.get_pt_vel(n, IF, 1)
        v_coord_2 = self.get_pt_vel(n, IF, 2)
        return v_coord_0.diff(t)*IF.x + v_coord_1.diff(t)*IF.y + v_coord_2.diff(t)*IF.z


    def get_omega(self, n, frame, coord):
        '''
        coord is 0..2 (x,y,z-direction)
        '''
        omega = self.body_frames[n].get_omega(frame)
        if coord == 0:
            v = frame.x
        elif coord == 1:
            v = frame.y
        elif coord == 2:
            v = frame.z
        omega_c = omega.express(frame, variables = True).dot(v)
        return omega_c.subs(self.kindiff_dict)

#    def subs_kindiff(self):
#        try:
#            self.d_c1=self.d_c1.subs(self.kindiff_dict)
#            self.d_c2=self.d_c2.subs(self.kindiff_dict)
#        except:
#            pass

    def correct_the_initial_state(self, m, x0):
        global IF, O, g, t
        #correct the initial state vector, dynamic var number m
        #TODO make it more general
        dynamic = self.q_flat + self.u_flat
        x00 = dict(zip(dynamic, x0))
        self.x00 = x00
        x,y = symbols('x y')
        if len(self.q[m]) == 1:
            x00[self.q[m][0]] = x
            equ = (self.d_c1.subs(self.const_dict)).subs(x00)
            x0[m] = sp_solve(equ)[0]
            x00 = dict(zip(dynamic, x0))
            m2 = m+self.dof
            x00[self.u[m][0]] = x
            equ = (self.d_c2.subs(self.const_dict)).subs(x00)
            x0[m2] = sp_solve(equ)[0]
        else:
            x00[self.q[m][0]] = 0.5
            x00[self.q[m][1]] = y
            self.equ_a = (self.d_c1.subs(self.const_dict)).subs(x00)
            x0[m] = 0.5
            x0[m+1] = sp_solve(self.equ_a,y)[0]

        return x0

    def set_const_dict(self, const_dict):
        self.const_dict = const_dict

    def kaneify_lin(self, q_ind, u_ind, q_dep, u_dep, c_cons, u_cons, x_op, a_op):
        global IF, O, g, t
        self.q_flat = q_ind #[ii for mi in self.q for ii in mi]
        self.u_flat = u_ind #[ii for mi in self.u for ii in mi]
        self.a_flat = [ii.diff() for ii in u_ind] #[ii for mi in self.a for ii in mi]
        #add external forces to the dynamic vector
        for oo in self.param_obj:
            self.parameters += oo.get_paras()
            self.parameters_diff.update(oo.get_diff_dict())
        self.freedoms = self.q_flat + self.u_flat + self.parameters
        self.dynamic = self.q_flat + self.u_flat + [t]  + self.f_ext_act + \
                       self.parameters  + self.f_int_act + self.f_t_models_sym

        self.kane = KanesMethod(IF, q_ind=self.q_flat, u_ind=self.u_flat,
                                q_dependent = q_dep, u_dependent = u_dep, configuration_constraints = c_cons,
                                velocity_constraints = u_cons, kd_eqs=self.kindiffs)
        self.fr, self.frstar = self.kane.kanes_equations(self.forces+self.torques, self.particles)
        #print u_cons
        self.A, self.B, self.inp_vec = self.kane.linearize(op_point=x_op, A_and_B=True,
                                     new_method=True, simplify =True)

        self.A = self.A.subs(self.kindiff_dict)
        self.B = self.B.subs(self.kindiff_dict)

        a_cons = [ ii.diff() for ii in u_cons]

        #too special TODO generalize...??
        for equ in u_cons:
            x = dynamicsymbols('x')
            for u in u_dep:
                x = sp_solve(equ, u)
                if len(x)>0:
                    self.A = self.A.subs({u:x[0]})
                    self.B = self.B.subs({u:x[0]})

        for equ in a_cons:
            x = dynamicsymbols('x')
            for u in u_dep:
                x = sp_solve(equ, u.diff())
                if len(x)>0:
                    self.A = self.A.subs({u.diff():x[0]})
                    self.B = self.B.subs({u.diff():x[0]})
        #generalize doit()
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                self.A[i,j] = self.A[i,j].doit()


        self.A = self.A.subs(self.accdiff_dict)
        self.A = self.A.subs(x_op)
        self.A = self.A.subs(a_op)
        self.B = self.B.subs(self.accdiff_dict)
        self.B.simplify()
        self.A.simplify()

        self.A = self.A.subs(self.const_dict)
        self.ev = self.A.eigenvals()
        k = 1
        myEV = []
        print( "*****************************" )
        for e in self.ev.keys():
            myEV.append(e.evalf())
            print( k, ". EV: ", e.evalf() )
            k+= 1
        print( "*****************************" )
        ar = self.matrix_to_array(self.A, self.A.shape[0])
        self.eig = eig(ar)

    def matrix_to_array(self, A, n):
        out = array(float(A[0,0]))
        for i in range(n-1):
            out = hstack((out, float(A[0,i+1])))
        for j in range(n-1):
            line = array(float(A[j+1,0]))
            for i in range(n-1):
                line = hstack((line, float(A[j+1,i+1])))
            out = vstack((out, line))
        print( out )
        return out

    def kaneify(self, simplify = False):
        global IF, O, g, t
        print( "Assemble the equations of motion ..." )
        tic = time.clock()
        self.q_flat = [ii for mi in self.q for ii in mi]
        self.u_flat = [ii for mi in self.u for ii in mi]
        self.a_flat = [ii for mi in self.a for ii in mi]
        #add external, internal forces to the dynamic vector + model forces + parameters
        for oo in self.param_obj:
            self.parameters += oo.get_paras()
            self.parameters_diff.update(oo.get_diff_dict())
        self.freedoms = self.q_flat + self.u_flat + [t] + self.parameters
        self.dynamic = self.q_flat + self.u_flat + [t] + self.f_ext_act + \
                        self.parameters + self.f_int_act + self.f_t_models_sym

        #we need the accdiff_dict for forces and rod forces
        for ii in range(len(self.u_flat)):
            x = dynamicsymbols('x')
            x = sp_solve(self.accdiff[ii],self.u_flat[ii].diff(t))[0]
            self.accdiff_dict.update({self.u_flat[ii].diff(t): x })
            #strange occurrence of - sign in the linearized version
            self.accdiff_dict.update({self.u_flat[ii].diff(t).subs({self.u_flat[ii]:-self.u_flat[ii]}): -x })


        if self.connect and not self.db_setup:
            print( "from the db ..." )
            wd = worldData()
            wd.put_str(self.mydb.get(self.name)[1])
            newWorld = list_to_world( wd.get_list() )
            self.M = newWorld.M
            self.F = newWorld.F
            self.kindiff_dict = newWorld.kindiff_dict
            if not self.dynamic == newWorld.dynamic:
                print( self.dynamic )
                print( newWorld.dynamic )
                raise Exception
        else:
            print( "calc further (subs)..." )
            self.kane = KanesMethod(IF, q_ind=self.q_flat, u_ind=self.u_flat, kd_eqs=self.kindiffs)
            self.fr, self.frstar = self.kane.kanes_equations(self.forces+self.torques, self.particles)
            self.kindiff_dict = self.kane.kindiffdict()
            self.M = self.kane.mass_matrix_full.subs(self.kindiff_dict)   # Substitute into the mass matrix

            self.F = self.kane.forcing_full.subs(self.kindiff_dict)       # Substitute into the forcing vector

            ##########################################################
            self.M = self.M.subs(self.parameters_diff)
            self.F = self.F.subs(self.parameters_diff)
            self.M = self.M.subs(self.const_dict)
            self.F = self.F.subs(self.const_dict)
        ######################################################
            if len(self.n_constr) > 0:
                self.F = self.F.subs(self.accdiff_dict)
                i_off = len(self.F)/2
                for ii in range(i_off):
                    for line in range(i_off):
                        jj = line + i_off
                        pxx = Poly(self.F[jj], self.a_flat[ii])
                        if len(pxx.coeffs()) > 1:
                            self.cxx = pxx.coeffs()[0]
                        else:
                            self.cxx = 0.
                        self.M[jj, ii+i_off] -= self.cxx
                accdiff_zero = {} #
                for k in self.a_flat:
                    accdiff_zero.update({ k:0 })
                self.F = self.F.subs(accdiff_zero)
        ########################################################
        # db stuff
        if self.connect and self.db_setup:
            wd = worldData(self)
            self.mydb.put(self.name, wd.get_str())
        ########################################################
        if simplify:
            print( "start simplify ..." )
            self.M.simplify()
            self.F.simplify()
        ########################################################
        print( "equations now in ram... lambdify the M,F parts" )
        self.M_func = lambdify(self.dynamic, self.M)               # Create a callable function to evaluate the mass matrix
        self.F_func = lambdify(self.dynamic, self.F)               # Create a callable function to evaluate the forcing vector
        #lambdify the part forces (only with self.freedom)
        for expr in self.f_int_expr:
            expr = expr.subs(self.kindiff_dict)
            self.f_int_lamb.append(lambdify(self.q_flat + self.u_flat, expr))
        ########################################################
        #lambdify the models, include also some dicts
        for model in self.models:
            model.set_subs_dicts([self.kindiff_dict, self.parameters_diff])
            model.lambdify_trafo(self.freedoms)
            self.f_models_lamb.append(model.force_lam)
        for expr in self.control_signals_expr:
            expr_ = expr.subs(self.kindiff_dict)
            self.control_signals_lamb.append(lambdify(self.q_flat + self.u_flat,expr_))

        nums = self.bodies.values()
        nums.sort()
        for n in nums[:-1]:
            self.body_list_sorted.append([oo for oo in self.bodies_obj.values() if oo.get_n() == n][0])
        toc = time.clock()
        #######################################################
        # set all dicts to all frames
        # d = [self.kindiff_dict, self.accdiff_dict, self.const_dict]
#        for oo in self.bodies_obj.values():
#            oo.set_dicts(d)
#            oo.get_frame().set_dicts(d)
#        for oo in self.marker_obj.values():
#            #oo.set_dicts(d)
#            oo.get_frame().set_dicts(d)
#        for oo in self.marker_fixed_obj.values():
#            #oo.set_dicts(d)
#            oo.get_frame().set_dicts(d)

        print( "finished ... ", toc-tic )


    def right_hand_side(self, x, t, args = []):
        #for filling up order of f_t see kaneify self.dynamics ...
        para = [ pf(t) for oo in self.param_obj for pf in oo.get_func()]
        r_int = [r(*x) for r in self.f_int_lamb]
        f_int = [self.f_int_func[i](r_int[i]) for i in range(len(r_int))]
        f_t = [t] + [fe(t) for fe in self.f_ext_func] + para + f_int

        inp = hstack((x, [t] + para))
        for ii in range(self.forces_models_n):
            F_T_model, signals = self.f_models_lamb[ii](inp)
            f_t += F_T_model
        #generate the control signals (to control somewhere else)
        for ii in range(len(self.control_signals_lamb)):
            self.control_signals[ii] = self.control_signals_lamb[ii](*x)
        #checkpoint output
        if t>self.tau_check:
            self.tau_check+=0.1
            print( t )

        arguments = hstack((x,f_t))       # States, input, and parameters
        #lu = factorized(self.M_func(*arguments))
        #dx = lu(self.F_func(*arguments)).T[0]

        dx = array(np_solve(self.M_func(*arguments),self.F_func(*arguments))).T[0]
        return dx

    def get_control_signal(self, no):
        try:
            return self.control_signals[no]
        except:
            return 0.

    def right_hand_side_ode(self, t ,x ):
        para = [ pf(t) for oo in self.param_obj for pf in oo.get_func()]
        f_t = [t] + [fe(t) for fe in self.f_ext_func] + para + \
        [fi(*x) for fi in f_int_lamb]
        inp = hstack((x, [t]+para))
        for ii in range(self.forces_models_n):
            F_T_model, signals = self.f_models_lamb[ii](inp)
            f_t += F_T_model

        arguments = hstack((x,f_t))       # States, input, and parameters
        dx = array(sc_solve(self.M_func(*arguments),self.F_func(*arguments))).T[0]
        return dx

    def res_body_pos_IF(self):
        global IF, O, g, t
        IF_coords = []
        for oo in self.body_list_sorted:
            IF_coords.append( oo.x().subs(self.const_dict) ) #self.get_pt_pos(ii,IF,0).subs(self.const_dict))
            IF_coords.append( oo.y().subs(self.const_dict) ) #self.get_pt_pos(ii,IF,1).subs(self.const_dict))
            IF_coords.append( oo.z().subs(self.const_dict) ) #self.get_pt_pos(ii,IF,2).subs(self.const_dict))
        f_t = [t] + self.parameters
        self.pos_cartesians_lambda = lambdify(self.q_flat+f_t, IF_coords)


    def res_body_orient(self):
        frame_coords = []
        for oo in self.body_list_sorted:
            #print "add orient for body: ",ii
            N_fixed = oo.get_frame()
            ex_x = N_fixed.x.dot(IF.x)
            ex_y = N_fixed.x.dot(IF.y)
            ex_z = N_fixed.x.dot(IF.z)
            ey_x = N_fixed.y.dot(IF.x)
            ey_y = N_fixed.y.dot(IF.y)
            ey_z = N_fixed.y.dot(IF.z)
            ez_x = N_fixed.z.dot(IF.x)
            ez_y = N_fixed.z.dot(IF.y)
            ez_z = N_fixed.z.dot(IF.z)
            frame_coords += [ex_x,ex_y,ex_z,ey_x,ey_y,ey_z,ez_x,ez_y,ez_z]
        f_t = [t] + self.parameters
        #self.frame_coords = lambdify(self.q_flat+[t], frame_coords)
        self.orient_cartesians_lambda = lambdify(self.q_flat+f_t, frame_coords)

    def res_fixed_body_frames(self, body):
        N_fixed = body.get_frame()
        frame_coords = []
        frame_coords.append( body.x().subs(self.const_dict))
        frame_coords.append( body.y().subs(self.const_dict))
        frame_coords.append( body.z().subs(self.const_dict))
        ex_x = N_fixed.x.dot(IF.x)
        ex_y = N_fixed.x.dot(IF.y)
        ex_z = N_fixed.x.dot(IF.z)
        ey_x = N_fixed.y.dot(IF.x)
        ey_y = N_fixed.y.dot(IF.y)
        ey_z = N_fixed.y.dot(IF.z)
        ez_x = N_fixed.z.dot(IF.x)
        ez_y = N_fixed.z.dot(IF.y)
        ez_z = N_fixed.z.dot(IF.z)
        frame_coords = frame_coords + [ex_x,ex_y,ex_z,ey_x,ey_y,ey_z,ez_x,ez_y,ez_z]
        f_t = [t] + self.parameters
        return lambdify(self.q_flat+f_t, frame_coords)

    def res_fixed_marker_frames(self, oo):
        N_fixed = oo.get_frame()
        frame_coords = []
        frame_coords.append( oo.x().subs(self.const_dict))
        frame_coords.append( oo.y().subs(self.const_dict))
        frame_coords.append( oo.z().subs(self.const_dict))
        ex_x = N_fixed.x.dot(IF.x)
        ex_y = N_fixed.x.dot(IF.y)
        ex_z = N_fixed.x.dot(IF.z)
        ey_x = N_fixed.y.dot(IF.x)
        ey_y = N_fixed.y.dot(IF.y)
        ey_z = N_fixed.y.dot(IF.z)
        ez_x = N_fixed.z.dot(IF.x)
        ez_y = N_fixed.z.dot(IF.y)
        ez_z = N_fixed.z.dot(IF.z)
        frame_coords = frame_coords + [ex_x,ex_y,ex_z,ey_x,ey_y,ey_z,ez_x,ez_y,ez_z]
        f_t = [t] + self.parameters
        return lambdify(self.q_flat+f_t, frame_coords)

    def res_body_marker_pos_IF(self):
        global IF, O, g, t
        IF_coords = []
        for oo in self.body_list_sorted:
            N_att = oo.get_N_att()
            x_att = N_att.px()
            y_att = N_att.py()
            z_att = N_att.pz()
            IF_coords.append( x_att.subs(self.const_dict))
            IF_coords.append( y_att.subs(self.const_dict))
            IF_coords.append( z_att.subs(self.const_dict))
            IF_coords.append( oo.x().subs(self.const_dict) )
            IF_coords.append( oo.y().subs(self.const_dict) )
            IF_coords.append( oo.z().subs(self.const_dict) )
        f_t = [t] + self.parameters
        self.connections_cartesians_lambda = lambdify(self.q_flat+f_t, IF_coords)

    def calc_acc(self):
        t = self.time
        u = self.x_t[:,self.dof:self.dof*2]
        self.acc = zeros(self.dof)
        for ti in range(1,len(t)):
            acc_line = []
            for ii in range(self.dof):
                acc_line = hstack((acc_line,(u[ti][ii]-u[ti-1][ii])/(t[ti]-t[ti-1])))
            self.acc = vstack((self.acc, acc_line))

    def res_rod_forces(self):
        self.f_rod = []
        for oo in self.body_list_sorted:
            N_fixed_n = oo.get_frame()
            Pt_n = oo.Pt()
            ay = self.get_pt_acc(ii,N_fixed_n,1).subs(self.const_dict)
            f_ex_constr = 0.
            for jj in self.forces:
                if jj[0] == Pt_n:
                    #print type(f_ex_constr)
                    f_ex_constr += jj[1].dot(N_fixed_n.y) #here assumed that the rod is in y-direction
            self.f_rod.append(oo.get_mass()*ay-f_ex_constr)
            #print "f_rod: ",self.f_rod[ii]
        self.rod_f_lambda = lambdify(self.q_flat+self.u_flat+self.a_flat, self.f_rod)

    def res_total_force(self, oo):
        global IF, O, g, t
        res_force = []
        res_force.append( oo.x().subs(self.const_dict) )
        res_force.append( oo.y().subs(self.const_dict) )
        res_force.append( oo.z().subs(self.const_dict) )
        n = oo.get_n()
        res_force.append(self.get_pt_acc(n,IF,0).subs(self.const_dict)*self.m[n])
        res_force.append(self.get_pt_acc(n,IF,1).subs(self.const_dict)*self.m[n])
        res_force.append(self.get_pt_acc(n,IF,2).subs(self.const_dict)*self.m[n])

        f_t = [t] + self.parameters
        return lambdify(self.q_flat+self.u_flat+self.a_flat+f_t, res_force)


    def res_kin_energy(self):
        E = 0
        #translatory
        #for ii in range(n_body+1):
        for oo in self.body_list_sorted:
            N_fixed = oo.get_frame() #self.body_frames[ii]
            vel =  oo.get_vel().subs(self.kindiff_dict)
            E += 0.5*oo.get_mass()*vel.magnitude()**2
        #substitude the parameter diffs
        E = E.subs(self.parameters_diff)
        f_t = [t] + self.parameters
        self.e_kin_lambda = lambdify(self.q_flat+self.u_flat+f_t, E)

    def res_speed(self):
        global IF, O, g, t
        N_fixed = self.body_frames[0]
        vel = N_fixed.get_vel_vec_IF().subs(self.kindiff_dict)
        vel_lateral = vel.dot(IF.x)*IF.x+vel.dot(IF.z)*IF.z
        vel_mag = vel_lateral.magnitude()
        vel_mag = vel_mag.subs(self.parameters_diff)
        f_t = [t] + self.parameters
        self.speed_lambda = lambdify(self.q_flat+self.u_flat+f_t, vel_mag)

    def res_rot_energy(self):
        E = 0
        #translatory
        for oo in self.body_list_sorted:
            N_fixed = oo.get_frame()
            omega = N_fixed.get_omega(IF).subs(self.kindiff_dict)
            omega_x = omega.dot(N_fixed.x)
            omega_y = omega.dot(N_fixed.y)
            omega_z = omega.dot(N_fixed.z)
            E += 0.5*(oo.I[0]*omega_x**2+oo.I[1]*omega_y**2+oo.I[2]*omega_z**2)
            #E += 0.5*(self.Ixx[ii]*omega_x**2+self.Iyy[ii]*omega_y**2+self.Izz[ii]*omega_z**2)
        #substitude the parameter diffs
        E = E.subs(self.parameters_diff)
        f_t = [t] + self.parameters
        self.e_rot_lambda = lambdify(self.q_flat+self.u_flat+f_t, E)

    def res_pot_energy(self):
        E = 0
        for x in self.pot_energy_saver:
            E += x.subs(self.const_dict)
        #substitude the parameter diffs
        if len(self.pot_energy_saver) > 0:
            E = E.subs(self.parameters_diff)
        f_t = [t] + self.parameters
        self.e_pot_lambda = lambdify(self.q_flat+f_t, E)

    #def res_signal(self):


    def show_figures(self):
        pass

    def prep_lambdas(self, moving_frames_in_graphics = [], fixed_frames_in_graphics = [], forces_in_graphics = [], bodies_in_graphics = {}):
        print( "start preparing lambdas..." )
        start = time.clock()
        self.res_body_pos_IF()
        self.res_body_orient()
        self.vis_frame_coords = []
        self.vis_fixed_frames = []
        self.vis_force_coords = []
        self.res_body_marker_pos_IF()
        self.res_kin_energy()
        self.res_pot_energy()
        self.res_rot_energy()
        self.res_speed()


        for str_m_b in moving_frames_in_graphics:
            n, N_fixed_n, is_body, oo = self._interpretation_of_str_m_b(str_m_b)
            if is_body:
                self.vis_frame_coords.append(self.res_fixed_body_frames(oo))
            else:
                self.vis_frame_coords.append(self.res_fixed_marker_frames(oo))

        for str_m_b in fixed_frames_in_graphics:
            n, N_fixed_n, is_body, body = self._interpretation_of_str_m_b(str_m_b)
            if not is_body:
                self.vis_fixed_frames.append(N_fixed_n)

        for str_m_b in forces_in_graphics:
            n, N_fixed_n, is_body, body = self._interpretation_of_str_m_b(str_m_b)
            if is_body:
                self.vis_force_coords.append(self.res_total_force(body))

        end = time.clock()

        for k,v in bodies_in_graphics.iteritems():
            n, N_fixed_n, is_body, body = self._interpretation_of_str_m_b(k)
            self.bodies_in_graphics.update({n:v})
        print( "finished ...",end-start )

    def prepare(self, path, save=True):
        #transform back to produce a state vector in IF
        n_body = self.n_body
        self.state = hstack(zeros((n_body+1)*3)) # 3 includes 3d cartesians + 1 time
        self.orient = hstack(zeros((n_body+1)*9)) #3 cartesians vectors e_x, e_y, e_z
        self.con = hstack(zeros((n_body+1)*6)) # 3d cartesian vector from-to (info)
        self.vis_body_frames = []
        for n in range(len(self.vis_frame_coords)):
            self.vis_body_frames.append(hstack(zeros(12))) #1 frame moving
        self.vis_forces = []
        for n in range(len(self.vis_force_coords)):
            self.vis_forces.append(hstack(zeros(6))) # 1 force on body
        self.e_kin = []
        self.e_pot = []
        self.e_tot = []
        self.e_rot = []
        self.speed = []
        self.signals = {}
        self.calc_acc()

        for ii in range(self.forces_models_n):
            self.signals.update({ii: zeros(self.models[ii].get_signal_length())})

        for ii in range(len(self.time)):
            tau = self.time[ii]
            f_t = [tau] + [ pf(tau) for oo in self.param_obj for pf in oo.get_func()]
            #controll-signals:
            # ???

            x_act = hstack((self.x_t[ii,0:self.dof], f_t))
            x_u_act = hstack((self.x_t[ii,0:self.dof*2], f_t))
            x_u_a_act = hstack((self.x_t[ii,0:self.dof*2], self.acc[ii], f_t))

            vx = self.pos_cartesians_lambda(*x_act)        #transports x,y,z
            orient = self.orient_cartesians_lambda(*x_act) #transports e_x und e_y
            vc = self.connections_cartesians_lambda(*x_act)
            for n in range(len(self.vis_frame_coords)):
                vf = self.vis_frame_coords[n](*x_act)
                self.vis_body_frames[n] = vstack((self.vis_body_frames[n], vf))
            for n in range(len(self.vis_force_coords)):
                fg = self.vis_force_coords[n](*x_u_a_act)
                self.vis_forces[n] = vstack((self.vis_forces[n], fg))

            speed = self.speed_lambda(*x_u_act)
            e_kin = self.e_kin_lambda(*x_u_act)
            e_rot = self.e_rot_lambda(*x_u_act)
            e_pot = self.e_pot_lambda(*x_act)

            for ii in range(self.forces_models_n):
                F_T_model, out_signals = self.f_models_lamb[ii](x_u_act)
                self.signals[ii] = vstack((self.signals[ii],out_signals))


            self.state = vstack((self.state,vx))
            self.orient = vstack((self.orient,orient))

            self.con = vstack((self.con, vc))

            self.e_rot.append(e_rot)
            self.e_kin.append(e_kin)
            self.e_pot.append(e_pot)
            self.e_tot.append(e_kin+e_pot+e_rot)
            self.speed.append(speed)
        for ii in range(self.forces_models_n):
            self.signals[ii] = self.signals[ii][1:]

        if save and not no_pandas:
            # currently only saving is supported....
            if self.name == '':
                store_filename = path+'/data.h5'
            else:
                store_filename = path+'/'+self.name+'.h5'
            self.store = pd.HDFStore(store_filename,complevel=2, complib='zlib')
            self.store['state'] = pd.DataFrame(self.state[:,:3],columns=['x', 'y', 'z']) # 3 includes 3d cartesians
            self.store['orient'] = pd.DataFrame(self.orient[:,:9],columns=['ex_x', 'ex_y', 'ex_z', 'ey_x', 'ey_y', 'ey_z', 'ez_x', 'ez_y', 'ez_z']) #2 cartesians vectors e_x, e_y
            self.store['con']  = pd.DataFrame(self.con)  # 3d cartesian vector from-to (info)
            #here we must consider on how to store the data properly...
            #self.store['vis_body_frames'] = pd.DataFrame(self.vis_body_frames) #1 frame moving
            #self.store['vis_forces'] = pd.DataFrame(self.vis_forces) # 1 force on body
            #self.store['vis_frame_coords'] = pd.DataFrame(self.vis_frame_coords)
            #self.store['vis_force_coords'] = pd.DataFrame(self.vis_force_coords)
            self.store['e_kin'] = pd.DataFrame(self.e_kin)
            self.store['time_'] = pd.DataFrame(self.time)
            self.store['x_t'] = pd.DataFrame(self.x_t)
            self.store['acc'] = pd.DataFrame(self.acc)
            self.store['e_pot'] = pd.DataFrame(self.e_pot)
            self.store['e_tot'] = pd.DataFrame(self.e_tot)
            self.store['e_rot'] = pd.DataFrame(self.e_rot)
            self.store['speed'] = pd.DataFrame(self.speed)
            for ii in range(self.forces_models_n):
                self.store['signals_'+str(ii)] = pd.DataFrame(self.signals[ii])
            #self.store['signals'] = pd.Series(self.signals)
            # the load function must set up a mubodyn world object sufficient for animate()...

    def plotting(self, t_max, dt, plots = 'standard'):
                #plotting
        if plots == 'standard':
            n = len(self.q_flat)
            n_max = int(t_max/dt)-2
            plt.subplot(2, 1, 1)
            lines = plt.plot(self.time[0:n_max], self.x_t[0:n_max, :n])
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(self.dynamic[:n])

            plt.subplot(2, 1, 2)
            lines = plt.plot(self.time, self.e_kin,self.time,self.e_rot,self.time, self.e_pot,self.time, self.e_tot)
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(['E_kin','E_rot', 'E_pot', 'E_full'])
            plt.show()
            #
        elif plots == 'y-pos':
            n = len(self.q_flat)
            n_max = int(t_max/dt)-2
            plt.subplot(2, 1, 1)
            lines = plt.plot(self.time[0:n_max], self.state[0:n_max, 4])
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(["y-Pos."])

            plt.subplot(2, 1, 2)
            lines = plt.plot(self.time, self.e_kin,self.time,self.e_rot,self.time, self.e_pot,self.time, self.e_tot)
            lab = plt.xlabel('Time [sec]')
            leg = plt.legend(['E_kin','E_rot', 'E_pot', 'E_full'])
            plt.show()

        elif plots == 'tire':

            plt.subplot(5, 1, 1)
            lines = plt.plot(self.time, array(self.signals[0])[:,0], self.time, array(self.signals[0])[:,2])
            #lab = plt.xlabel('Time [sec]')
            leg1 = plt.legend(['Fx [N]', 'Fz [N]'])

            plt.subplot(5, 1, 2)
            lines = plt.plot(self.time, array(self.signals[0])[:,1])
            #lab = plt.xlabel('Time [sec]')
            leg1 = plt.legend(['Fy [N]'])

            plt.subplot(5, 1, 3)
            lines = plt.plot( self.time, array(self.signals[0])[:,3])
            #lab = plt.xlabel('Time [sec]')
            leg2 = plt.legend(['Tz [Nm]'])

            plt.subplot(5, 1, 4)
            lines = plt.plot( self.time, array(self.signals[0])[:,4])
            lab = plt.xlabel('Time [sec]')
            leg3 = plt.legend(['Slip [%]'])

            plt.subplot(5, 1, 5)
            lines = plt.plot( self.time, array(self.signals[0])[:,5])
            lab = plt.xlabel('Time [sec]')
            leg4 = plt.legend(['Alpha [grad]'])
            plt.show()

    def animate(self, t_max, dt, scale = 4, time_scale = 1, t_ani = 30., labels = False, center = -1, f_scale = 0.1, f_min = 0.2, f_max = 5.):
        #stationary vectors:
        a = animation(scale)
        for fr in self.vis_fixed_frames:
            a.set_stationary_frame(fr)
        for n in range(len(self.vis_frame_coords)):
            a.set_dynamic_frame(self.vis_body_frames[n])
        for n in range(len(self.vis_force_coords)):
            a.set_force(self.vis_forces[n], f_scale, f_min, f_max)

        a = a.s_animation(self.state, self.orient, self.con, self.con_type, self.bodies_in_graphics, self.speed, dt, t_ani, time_scale, scale, labels = labels, center = center)
        return a

    def prepare_integrator_pp(self, x0, delta_t):
        self.ode15s = ode(self.right_hand_side_ode)
        self.ode15s.set_integrator('lsoda', method='lsoda', min_step = 1e-6, atol = 1e-6, rtol = 1e-5, with_jacobian=False)
        self.ode15s.set_initial_value(x0, 0.)
        self.delta_t = delta_t
    def inte_grate_pp(self):
        self.ode15s.integrate(self.ode15s.t+self.delta_t)
        return self.ode15s.y, self.ode15s.t

    def inte_grate_full(self, x0, t_max, delta_t, mode = 0, tolerance = 1.0):
        global IF, O, g, t

        self.time = linspace(0, t_max, int(t_max/delta_t))
        print( "start integration ..." )
        start = time.clock()
        ###
        #some int stuff
        if mode == 1:
            ode15s = ode(self.right_hand_side_ode)
            ode15s.set_integrator('lsoda', min_step = 1e-6, atol = 1e-6, rtol = 1e-7, with_jacobian=False)
            #method = 'bdf'
            ode15s.set_initial_value(x0, 0.)
            self.x_t = x0
            while ode15s.t < t_max:
                ode15s.integrate(ode15s.t+delta_t)
                self.x_t = vstack((self.x_t,ode15s.y))
        elif mode == 0:
            self.x_t = odeint(self.right_hand_side, x0, self.time, args=([0.,0.],) , hmax = 1.0e-1, hmin = 1.0e-7*tolerance, atol = 1e-5*tolerance, rtol = 1e-5*tolerance, mxords = 4, mxordn = 8)

        end = time.clock()
        print( "end integration ...", end-start )




    def constr_lin(self, x_op, quad = False):
        n_const = len(self.eq_constr)
        dofs = range(self.dof)
        c_cons = []
        u_cons = []

        for i in range(n_const):
            self.eq_constr[i] = self.eq_constr[i].subs(self.kindiff_dict)
            lam = []
            #lam_lin = []
            equ1a = symbols('equ1a')
            equ1a = 0
            for d in dofs:
                equ = self.eq_constr[i]
                #lam.append(equ)
                term = equ.subs(x_op)+equ.diff(self.q_flat[d]).subs(x_op)*self.linfaktor(x_op, self.q_flat[d])
                if quad:
                    term += 0.5* equ.diff(self.q_flat[d]).diff(self.q_flat[d]).subs(x_op)*(self.linfaktor(x_op, self.q_flat[d]))**2
                lam.append(term)
                equ1a += lam[-1].simplify()
            c_cons.append(equ1a)
            u_cons.append(equ1a.diff().subs(self.kindiff_dict))
        print( "c_cons: ", c_cons )
        return c_cons, u_cons

    def linearize(self, x_op, a_op, quad = False):
        """
        Function to prepare the linerarization process (kaneify_lin)
        """
        #construct the new constraint equ from previous constraint equations:
        n_const = len(self.eq_constr)
        dofs = range(self.dof)
        c_cons, u_cons = self.constr_lin(x_op, quad = quad)

        #try to find the dependent and independent variables
        q_ind = []
        u_ind = []
        q_dep = []
        u_dep = []
        for d in dofs:
            dep = False
            for eq in range(n_const):
                x = sp_solve(c_cons[eq], self.q_flat[d])
                if not len(x) == 0 and len(q_dep) < n_const:
                    q_dep.append(self.q_flat[d])
                    u_dep.append(self.u_flat[d])
                    dep = True
                    break
            if not dep:
                q_ind.append(self.q_flat[d])
                u_ind.append(self.u_flat[d])
        print( "ind: ",q_ind )
        print( "dep: ",q_dep )
        #repair the operation point to be consistent with the constraints
        # and calc the backtrafo
        self.equ_out = []
        repl = []
        n = n_const
        c_copy = copy.copy(c_cons)
        for q in q_dep:
            for eq in range(n):
                x = sp_solve(c_copy[eq], q)
                if not len(x) == 0:
                    repl.append({q:x[0]})
                    #print q, x[0]
                    self.equ_out.append(x[0])
                    c_copy.pop(eq)
                    n = n-1
                    break
            for eq in range(len(c_copy)):
                c_copy[eq]=c_copy[eq].subs(repl[-1])
        repl.reverse()
        for eq in range(n_const):
            for pair in repl:
                self.equ_out[eq] = self.equ_out[eq].subs(pair) #order is relevant


        self.mydict = dict(zip(q_dep,self.equ_out))
        self.q_flat_lin = [term.subs(self.mydict) for term in self.q_flat]

        self.back_trafo_q_all = lambdify( q_ind, self.q_flat_lin)

        q_inp = []
        for d in range(len(q_ind)):
            q_inp.append(q_ind[d].subs(x_op))

        q_z = self.back_trafo_q_all(*q_inp)
        self.x_op_new = dict(zip(self.q_flat, q_z))


        if n_const > 0:
            self.forces = self.forces[0:-n_const] #pop the extra constraint forces
        self.kaneify_lin(q_ind, u_ind, q_dep, u_dep, c_cons, u_cons, x_op, a_op)

    def calc_Jacobian(self, n):
        """
        Function to calculate the jacobian, after integration for calc-point no n.

        :param n: the list-number of the integrated state vector
        """
        t_op = self.time[n]
        x_op = self.x_t[n]
        eps = 1.0e-12
        f0 = self.right_hand_side(x_op, t_op)
        f_eps = []
        dim = range(len(f0))
        for n in dim:
            x_op[n] += eps
            f_eps.append(self.right_hand_side(x_op, t_op))
            x_op[n] -= eps
        jac = zeros(len(f0))
        for n in dim:
            line = []
            for m in dim:
                line = hstack((line,(f_eps[m][n]-f0[n])/eps))
            jac = vstack((jac, line))
        return jac[1:]

    def calc_lin_analysis_n(self, n):
        """
        Function to calculate linear anlysis (stability), after integration for calc-point no n.

        :param n: the list-number of the integrated state vector
        """
        jac = self.calc_Jacobian(n)
        ev = eig(jac)[0]
        print( "Eigenvalues: time [s] ",self.time[n] )
        for e in range(len(ev)):
            print( str(e)+"... ",ev[e] )
        return jac

    def linfaktor(self, x_op, q):
        """
        Returns the linear factor of a single variable at value q.

        :param x_op: the symbol
        :param q: the value of the zero of this linear-factor
        """
        x, x0 = symbols('x, x0')
        equ = x-x0
        equ1 = equ.subs({x0:q})
        equ1 = equ1.subs(x_op)
        equ1 = equ1.subs({x:q})
        return equ1

