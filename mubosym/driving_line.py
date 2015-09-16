# -*- coding: utf-8 -*-
"""
Code for driving line definition
================================

Created on Sat May 16 18:33:20 2015

@author: oliver
"""

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from numpy import sqrt, sin, cos, sign

import numpy as np
#import mubosym as mbs

def create_kl_2d(filename):
    radius = 40.0
    tau = np.linspace(-5.,40.,30)
    x = radius*sin(0.2*tau)
    y = radius*(1. - cos(0.2*tau))+0.5
    with open(filename, 'w') as f:
        for ii in range(len(tau)):
            f.write(str(tau[ii])+" "+ str(x[ii])+" "+str(y[ii]) + "\n")
        

def read_kl_2d(filename):
    with open(filename, 'r') as f:
        inp = f.read()
    inlist = inp.split('\n')
    inlist = [ x for x in inlist if x != '']
    inlist = [ x for x in inlist if x[0] != '#']
    inlist = [x.split(' ') for x in inlist]
    #print inlist
    tau = np.array([ float(x[0]) for x in inlist])
    x_in = np.array([ float(x[1]) for x in inlist])
    y_in = np.array([ float(x[2]) for x in inlist])
    return tau, x_in, y_in
    

class interp_dl(object):
    """
    The main connection between an external force characterized by a number of points and the mubosym
    After running the initialization the base-functions are setup (by means of optimized coefficients)
    
    :param filename: the external file with a list of x y - values (table, separation sign is space), if filename is empty the function f11 is taken instead
    :param tst: if true the result of the optimization is plotted 
    """
    def __init__(self, filename, tst = False):
        self.tau, self.vx, self.vy = read_kl_2d(filename)
        self.eps = 1e-6
        self.f_interp_x = interp1d(self.tau, self.vx, kind = 'cubic', bounds_error=False)
        self.f_interp_y = interp1d(self.tau, self.vy, kind = 'cubic', bounds_error=False)
        self.pt = (0.,0.)
        self.n = 101  
        self.out = (0.,0.)
        # Test:
        if tst:
            
            tau_dense = np.linspace(0., 15., 200)
            x_dense = []
            y_dense = []
            for t in tau_dense:
                x_dense.append(self.f_interp(t)[0])
                y_dense.append(self.f_interp(t)[1])            
            lines = plt.plot( x_dense, y_dense )
            plt.show()
            
    def f_interp(self, t):
        return (self.f_interp_x(t),self.f_interp_y(t))
    
    def tangential(self, t):
        eps = self.eps 
        return (self.f_interp_x(t+eps)-self.f_interp_x(t-eps))/(2.*eps),(self.f_interp_y(t+eps)-self.f_interp_y(t-eps))/(2.*eps)
        
    def distance(self, pt, t_start):
        self.pt = pt
        t_solve= fsolve(self.fct, t_start)
        v = self.tangential(t_solve)
        r = (self.f_interp_x(t_solve)-self.pt[0],self.f_interp_y(t_solve)-self.pt[1]) 
        s = sign(v[0]*r[1]-v[1]*r[0])[0]
        return ( t_solve[0], self.f_interp_x(t_solve)[0], self.f_interp_y(t_solve)[0], s*sqrt((self.f_interp(t_solve)[0]-self.pt[0])**2+(self.f_interp(t_solve)[1]-self.pt[1])**2)[0] )
        
    def fct(self, t):
        v = self.tangential(t)
        r = (self.f_interp_x(t)-self.pt[0],self.f_interp_y(t)-self.pt[1]) 
        return v[0]*r[0] + v[1]*r[1]
        
    def tang_diff(self, v_ext, t):
        v = self.tangential(t)
        return (v[0]*v_ext[1]-v[1]*v_ext[0])
        
if __name__ == "__main__":
    name = "/home/oliver/testgit/mubosym/data/line_01.dat"
    create_kl_2d(name)
    k = interp_dl(filename = name, tst=True)
    

