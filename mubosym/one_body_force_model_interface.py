# -*- coding: utf-8 -*-
"""
External model connection: general description
==============================================

Created on Wed May 27 18:02:53 2015

@author: oliver
"""
import sys
from sympy import lambdify, symbols

class one_body_force_model():
    '''
    a one body force model consists of:
    
    * coordinate trafo generalized coords -> body coordinates (denoted list of) including pos, vel, orientation, and omega 
    * force calculator given as a python function with input according to our interface
    * some preparation function: lambdifier to include symbolic functions into lambdas
    
    '''
    def __init__(self):
        # setup parameters
        self.D = 200000.
        self.gamma = 200.0
        self.y0 = -1.0
        self.trafo = []
        
        
    def set_coordinate_trafo(self, tr):
        """
        input function for the coordinate trafo expressions (sympy)
        """
        self.trafo = tr
        
    def set_kindiff_dict(self, kindiff_dict):
        for ii in range(len(self.trafo)):
            self.trafo[ii] = self.trafo[ii].subs(kindiff_dict)
        
    def lambdify_trafo(self, generalized_coords):
        """
        this is the core function to lambdify the coordinate trafos in general the trafos must be explicitely set via set_coordinate_trafo called from MBSCore (see therein)
        """
        if len(self.trafo) < 12:
            print("call set_coordinate_trafo first")
            sys.exit(0)
        t = symbols('t')
        self.lam_t = lambdify(generalized_coords, t)
        self.lam_x = lambdify(generalized_coords, self.trafo[0])
        self.lam_y = lambdify(generalized_coords, self.trafo[1])
        self.lam_z = lambdify(generalized_coords, self.trafo[2])
        self.lam_nx = lambdify(generalized_coords, self.trafo[3])
        self.lam_ny = lambdify(generalized_coords, self.trafo[4])
        self.lam_nz = lambdify(generalized_coords, self.trafo[5])
        self.lam_x_pt = lambdify(generalized_coords, self.trafo[6])
        self.lam_y_pt = lambdify(generalized_coords, self.trafo[7])
        self.lam_z_pt = lambdify(generalized_coords, self.trafo[8])
        self.lam_omega_x = lambdify(generalized_coords, self.trafo[9])
        self.lam_omega_y = lambdify(generalized_coords, self.trafo[10])
        self.lam_omega_z = lambdify(generalized_coords, self.trafo[11])
        
    def trafo_lam(self, w):
        """
        just for reference all coordinate trafos as lambdas (not used at the moment)
        """
        return [self.lam_x(*w), self.lam_y(*w), self.lam_z(*w), \
                self.lam_nx(*w), self.lam_ny(*w), self.lam_nz(*w), \
                self.lam_x_pt(*w), self.lam_y_pt(*w), self.lam_z_pt(*w), \
                self.lam_omega_x(*w), self.lam_omega_y(*w), self.lam_omega_z(*w)]
    
    def force_lam(self, w):
        """
        the model force/torque via lambdified expressions, input parameter here is always the full state vecor t,q,u
        Output is the force/toque via the model calc-function the nested input for the calc routine is fully possible written out:
        
        * self.lam_t, self.lam_x, self.lam_y, self.lam_z,
        * self.lam_nx, self.lam_ny, self.lam_nz, 
        * self.lam_x_pt, self.lam_y_pt, self.lam_z_pt, 
        * self.lam_omega_x self.lam_omega_y, self.lam_omega_z 
        
        but can be reduced to a subset
        """
        return self.calc([ self.lam_y(*w), self.lam_y_pt(*w) ] )
                
    def calc(self, inp):
        """
        the python function which connects some external model calculation with the mbs model e.g. tire-model, rail model
        
        * input list inp are some relevant model coordinates (out of 12 possible): [ x, y, z, nx, ny, nz, x_pt, y_pt, z_pt, omega_x, omega_y, omega_z ] = inp
        * output list is force in cartesian coord. world and torque cartesian coord. world
        
        """
        [ y , y_pt] = inp
        F_x = 0.
        if y<0:
            F_y = -self.D*(y-self.y0) - self.gamma*y_pt
        else:
            F_y = 0.
        F_z = 0.
        T_x = 0.
        T_y = 0.
        T_z = 0.
        
        return [F_x, F_y, F_z, T_x, T_y, T_z]
    