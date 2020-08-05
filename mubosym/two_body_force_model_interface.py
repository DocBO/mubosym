# -*- coding: utf-8 -*-
"""
External model connection: general description
==============================================

Created on Wed May 27 18:02:53 2015

@author: oliver
"""
import sys
from sympy import lambdify, symbols

class two_body_force_model():
    '''
    a one body force model consists of:

    * coordinate trafo generalized coords -> body coordinates (denoted list of) including pos, vel, orientation, and omega
    * force calculator given as a python function with input according to our interface
    * some preparation function: lambdifier to include symbolic functions into lambdas

    '''
    def __init__(self, para = []):
        # setup parameters
        if len(para) > 0:
            self.D = para[0]
            self.gamma = para[2]
            self.r0 = para[1]
        else:
            self.D = 20000.
            self.gamma = 500.0
            self.r0 = 0.9
        self.trafo = []
        self.signals = []
        self.signals_values = []


    def set_coordinate_trafo(self, tr):
        """
        input function for the coordinate trafo expressions (sympy) for two bodies, order see function lambdify_trafo
        """
        self.trafo = tr

    def set_kindiff_dict(self, kindiff_dict):
        for ii in range(len(self.trafo)):
            self.trafo[ii] = self.trafo[ii].subs(kindiff_dict)

    def set_subs_dicts(self, subs_dicts):
        for sd in subs_dicts:
            for ii in range(len(self.trafo)):
                self.trafo[ii] = self.trafo[ii].subs(sd)
            for ii in range(len(self.signals)):
                self.signals[ii] = self.signals[ii].subs(sd)


    def add_signal(self, expr):
        self.signals.append(expr)

    def lambdify_trafo(self, generalized_coords):
        """
        this is the core function to lambdify the coordinate trafos in general the trafos must be explicitely set via set_coordinate_trafo called from MBSCore (see therein)
        """
        if len(self.trafo) < 2:
            print("call set_coordinate_trafo first")
            sys.exit(0)
        t = symbols('t')
        print(generalized_coords)
        self.lam_t = lambdify(generalized_coords, t)
        self.lam_r = lambdify(generalized_coords, self.trafo[0])
        self.lam_r_pt = lambdify(generalized_coords, self.trafo[1])
        self.lam_signals = [ lambdify(generalized_coords, expr) for expr in self.signals]

    def trafo_lam(self, w):
        """
        just for reference all coordinate trafos as lambdas (not used at the moment)
        """
        return [self.lam_r(*w),self.lam_r_pt(*w)]

    def force_lam(self, w):
        """
        the model force/torque via lambdified expressions, input parameter here is always the full state vecor t,q,u
        Output is the force/toque via the model calc-function the nested input for the calc routine is fully possible written out:

        * self.lam_t, self.lam_r, self.lam_r_pt

        but can be reduced to a subset
        """
        self.signals_values = [x(*w) for x in self.lam_signals]
        return self._calc([ self.lam_r(*w), self.lam_r_pt(*w) ] )

    def _calc(self, inp):
        """
        the python function which connects some external model calculation with the mbs model e.g. tire-model, rail model

        * input list inp are some relevant model coordinates (out of 12 possible): [ x, y, z, nx, ny, nz, x_pt, y_pt, z_pt, omega_x, omega_y, omega_z ] = inp
        * output list is force in cartesian coord. world and torque cartesian coord. world

        """
        in_signals = self.signals_values
        #print in_signals
        [ r , r_pt] = inp
        F_r = self.D*(r-self.r0) + self.gamma*r_pt
        return [F_r],[r, r_pt]

    def get_signal_length(self):
        return 2
