# -*- coding: utf-8 -*-
"""
Code for use of Spline-Interpolation
====================================

Created on Sat May 16 18:33:20 2015

@author: oliver
"""

from sympy import symbols, lambdify, sign, re, acos, asin, sin, cos, bspline_basis
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import numpy as np


def read_kl(filename):
    with open(filename, 'r') as f:
        inp = f.read()
    inlist = inp.split('\n')
    inlist = [ x for x in inlist if x != '']
    inlist = [ x for x in inlist if x[0] != '#']
    inlist = [x.split(' ') for x in inlist]
    #print inlist
    x_in = np.array([ float(x[0]) for x in inlist])
    y_in = np.array([ float(x[1]) for x in inlist])
    return x_in, y_in
    

class interp(object):
    """
    The main connection between an external force characterized by a number of points and the mubosym
    After running the initialization the base-functions are setup (by means of optimized coefficients)
    
    :param filename: the external file with a list of x y - values (table, separation sign is space), if filename is empty the function f11 is taken instead
    :param tst: if true the result of the optimization is plotted 
    """
    def __init__(self, filename, tst = False):
        self.vx, self.vy = read_kl(filename)
        self.f_interp = interp1d(self.vx, self.vy, kind = 'linear', bounds_error=False)
        # Test:
        if tst:
            x_dense = np.linspace(-1., 15., 200)
            y_dense = []
            for xx in x_dense:
                y_dense.append(self.f_interp(xx))            
            lines = plt.plot( x_dense, y_dense )
            plt.show()
            

if __name__ == "__main__":
    k = interp(filename = "/home/oliver/python_work/mubosym01/mubosym/vel_01.dat", tst=True)
    

