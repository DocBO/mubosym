# -*- coding: utf-8 -*-
"""
simple_tire_model_interface
===========================
Created on Wed May 27 18:02:53 2015

@author: oliver
"""
import sys
from sympy import lambdify, symbols
import numpy as np

b      = [1.5,0.,1100.,0.,300.,0.,0.,0.,-2.,0.,0.,0.,0.,0.]
a      = [1.4,0.,1100.,1100.,10.,0.,0.,-2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]

def Pacejka_F_long(Fz, slip):
    """
    longitudinal force
    
    :param (float) Fz: Force in vertical direction in N
    :param (float) slip: relative slip fraction (0..1)
    """
    if Fz == 0:
        return 0.
    slip = slip*100.0
    Fz = Fz/1000.0
    C    = b[0]
    D    = Fz*(b[1]*Fz+b[2])
    BCD  = (Fz*(b[3]*Fz+b[4]))*np.exp(-b[5]*Fz)
    B    = BCD/(C*D)
    H    = b[9]*Fz+b[10]
    V    = b[11]*Fz+b[12]
    E    = ((b[6]*Fz*Fz)+b[7]*Fz+b[8])*(1-(b[13]*np.sign(slip+H)))
    Bx1  = B*(slip+H)
    Fx   = D*np.sin(C*np.arctan(Bx1-E*(Bx1-np.arctan(Bx1))))+V    
    return Fx

def Pacejka_F_lat(Fz, alpha, camber):
    """
    lateral force
    
    :param (float) Fz: Force in vertical direction in N
    :param (float) alpha: slip angle in rad
    :param (float) camber: camber angle in rad
    """
    if Fz == 0:
        return 0.
    alpha = alpha * 180.0/np.pi
    camber = camber * 180.0/np.pi
    Fz = Fz/1000.0
    C    = a[0]
    D    = Fz*(a[1]*Fz+a[2])*(1-a[15]*np.power(camber,2))
    BCD  = a[3]*np.sin(np.arctan(Fz/a[4])*2)*(1-a[5]*np.fabs(camber))
    B    = BCD/(C*D)
    H    = a[8]*Fz+a[9]+a[10]*camber
    V    = a[11]*Fz+a[12]+(a[13]*Fz+a[14])*camber*Fz
    E    = (a[6]*Fz+a[7])*(1-(a[16]*camber+a[17])*np.sign(alpha+H))
    Bx1  = B*(alpha+H)
    Fy   = D*np.sin(C*np.arctan(Bx1-E*(Bx1-np.arctan(Bx1))))+V
    return Fy


class simple_tire_model():
    """
    A one body force model consists of:
    
    * coordinate trafo generalized coords -> body coordinates (denoted list of) including pos, vel, orientation, and omega 
    * force calculator given as a python function with input according to our interface
    * some preparation function: lambdifier to include symbolic functions into lambdas
    
    """
    def __init__(self, paras = []):
        # setup parameters
        self.t = 0.
        self.D = 200000.
        self.gamma = 200.0
        self.y0 = 0.0
        self.C_side = 4500.0
        self.C_align = 200.0
        self.C_slip = 300.0 
        self.R_tire = 0.33
        self.trafo = []
        self.F_max = 4500.0
        self.gamma_torque = 2.0
        self.max_p = 100.0
        self.tau = 0.1
        self.signals = []
        self.signals_values = []
        
    def set_coordinate_trafo(self, tr):
        """
        Input function for the coordinate trafo expressions (sympy).
        
        :param tr: the transformation expressions as given in the mbs setup for the body
        """
        self.trafo = tr
        
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
        This is the core function to lambdify the coordinate trafos in general
        the trafos must be explicitely set via set_coordinate_trafo called from MBSCore (see therein)
        
        :param generalized_coords: the generalized coords (symbols) of the final mbs setup (called in kaneify)
        """
        if len(self.trafo) < 12:
            print("call set_coordinate_trafo first")
            sys.exit(0)
#        for ii in range(12):
#            print ii, self.trafo[ii]
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
        self.lam_signals = [ lambdify(generalized_coords, expr) for expr in self.signals]        
        
    def trafo_lam(self, w):
        """
        Just for reference all coordinate trafos as lambdas (not used at the moment).
        
        :param w: the generalized coords (float numbers) of the final mbs setup
        """
        return [self.lam_t(*w), self.lam_x(*w), self.lam_y(*w), self.lam_z(*w), \
                self.lam_nx(*w), self.lam_ny(*w), self.lam_nz(*w), \
                self.lam_x_pt(*w), self.lam_y_pt(*w), self.lam_z_pt(*w), \
                self.lam_omega_x(*w), self.lam_omega_y(*w), self.lam_omega_z(*w)]
    
    def force_lam(self, w):
        """
        The model force/torque via lambdified expressions, input parameter here is always the full state vecor t,q,u.
        Output is the force/toque via the model calc-function the nested input for the calc routine is fully possible written out:
        
        * self.lam_t, self.lam_x, self.lam_y, self.lam_z, 
        * self.lam_nx, self.lam_ny, self.lam_nz, 
        * self.lam_x_pt, self.lam_y_pt, self.lam_z_pt, 
        * self.lam_omega_x self.lam_omega_y, self.lam_omega_z 
        
        but can be reduced to a subset
        
        :param w: the generalized coords (float numbers) of the final mbs setup, The order has to be equal the one in calc.
        """
        self.signals_values = [x(*w) for x in self.lam_signals]
        return self._calc([ self.lam_t(*w), self.lam_y(*w), \
                           self.lam_x_pt(*w), self.lam_y_pt(*w), self.lam_z_pt(*w),\
                           self.lam_omega_z(*w) ] )
                
    def _calc(self, inp):
        """
        The python function which connects some external model calculation with the mbs model
        e.g. tire-model, rail model. It is only called internally by force_lam.
        
        * input list inp are some relevant model coordinates (out of 12 possible): [ x, y, z, nx, ny, nz, x_pt, y_pt, z_pt, omega_x, omega_y, omega_z ] = inp
        * output list is force in cartesian coord. world and torque cartesian coord. world
        
        :param inp: the subset of all possible coord. of one body (see list), here expected as float numbers. The order has to be equal the one in force_lam
        """
        signals = self.signals_values
        [ t, y , x_pt, y_pt, z_pt, omega_z ] = inp
        #print "SSSig: ",signals
        eps = 5.0e-1
        #preset values
        F_x = 0.
        F_y = 0.
        F_z = 0.
        T_x = 0.
        T_y = 0.
        T_z = 0.
        #vertical reaction force
        if y<0:
            F_y = -self.D*(y-self.y0) - self.gamma*y_pt
        else:
            F_y = 0.
        #side slip angle
        alpha = np.arctan2(z_pt,(x_pt+eps)) #in the tire carrier frame
        #slip
        slip = (omega_z * self.R_tire + x_pt)/np.abs(x_pt+eps)
        #######################################################
        # Pacejka - Model:
        F_z = - Pacejka_F_lat(F_y, alpha, 0.)
        F_x = - Pacejka_F_long(F_y, slip)
        T_z = F_x * self.R_tire - self.gamma_torque * omega_z
        #print F_y
        #self.oz += 1./10.*delta_t * T_z
        return [F_x, F_y, F_z, T_x, T_y, T_z], [F_x, F_y, F_z, T_z, 1e+2*slip, 180/np.pi*alpha]
    
    def get_signal_length(self):
        return 6
        

