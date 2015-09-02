# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:33:20 2015

@author: oliver
"""
from __future__ import print_function

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
    print( inlist )
    x_in = np.array([ float(x[0]) for x in inlist])
    y_in = np.array([ float(x[1]) for x in inlist])
    return x_in, y_in
    
def write_kl(filename, vx, vy):
    with open(filename, 'w') as f:
        for ii in range(len(vx)):
            f.write(str(vx[ii])+" "+str(vy[ii])+"\n")
    
    

def create_knots(a,b,n,p):
    """
    Function to create knots for b-spline basis startup. The distribution on the edges are crucial for the behaviour.
    
    :param a: the left edge
    :param b: the right edge
    :param n: the total number of knots
    :param p: the number of edge knots, the rest is distributed uniformly
    """
    eps = 1.0e-8
    p_star = p
    k_uni = np.linspace(a+p_star*eps, b-p_star*eps, n-2*p_star)
    knots = np.zeros(n)
    for i in range(p_star):
        knots[i] = a + i*eps
        knots[n-i-1] = b - i*eps
    
    knots[p_star:n-p_star] = k_uni
    return knots

def f11(xx):
    """
    Example of a analytic expression replacing the external point number
    
    :param xx: the distance between two bodies (or markers)
    """
    return 20.0/(0.5*xx*xx+1.0) #np.sqrt(np.abs(xx*xx))    

class characteristic_line():
    """
    The main connection between an external force characterized by a number of points and the mubosym
    After running the initialization the base-functions are setup (by means of optimized coefficients)
    
    :param filename: the external file with a list of x y - values (table, separation sign is space), if filename is empty the function f11 is taken instead
    :param tst: if true the result of the optimization is plotted 
    """
    def __init__(self, filename, tst = False):
        a = -1.  #left border limit
        b = 10. #right border limit
        self.n = n = 41
        self.p = p = 3
        
        #prepare vx, vy (can also be retrieved from an external file)
        vx = np.linspace(a, b, n)
        vy = []
        for xx in vx:
            vy.append(f11(xx)) 
        
        self.x = x = symbols('x')
        N = np.zeros(n-p-1)
        self.lam_base = [] # the lambdified basis expressions
        self.sym_base = [] # the symbolic basis expressions
        knots = create_knots(a,b,n,p)
        print( "Prepare Kennlinien Base-Fct..." )
        for k in range(n-p-1):
            #print k
            u = bspline_basis(p, knots, k, x)
            f = lambdify(x,u)
            self.lam_base.append(f)
            self.sym_base.append(u)
            
        for xx in vx:
            line = []
            for k in range(n-p-1):
                line = np.hstack((line, self.lam_base[k](xx)))
            N = np.vstack((N, line))
        N = N[1:]
            
        #    line = np.hstack(line, )
        Ntr = np.transpose(N)
        
        #solve the Gauss-Problem:
        NTN = np.dot(Ntr, N)
        NTN_inv = np.linalg.inv(NTN)
        Ps_inv = np.dot(NTN_inv, Ntr)
        self.C = np.dot(Ps_inv, vy)
        # Test:
        if tst:
            self.x_dense = np.linspace(a, b, 200)
            self.y_dense = []
            for xx in self.x_dense:
                self.y_dense.append(self.myfunc(xx))            
            lines = plt.plot(vx, vy, self.x_dense, self.y_dense )
            plt.show()
            
        self.gl = n-p-1
    def myfunc(self,xx):
        """
        The lambdified force expression after optimizing the coefficients
        
        :param xx: the distance between two bodies (or markers)
        """
        out = 0.
        for k in range(self.n-self.p-1):
            out += self.C[k]*self.lam_base[k](xx)
        return out

    def myfunc_0(self, arg):
        """
        The analytic force expression after optimizing the coefficients
        
        :param xx: the distance between two bodies (or markers)
        """
        out = 0.
        for k in range(self.gl):
            out -= self.C[k]*self.sym_base[k].subs({self.x:arg})
        return out
        


if __name__ == "__main__":
    k = characteristic_line(filename = "", tst=True)
    write_kl("./force_kl1.dat", k.x_dense, k.y_dense)

