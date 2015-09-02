# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:33:38 2015

@author: loc_vijay
"""
import numpy as np
import matplotlib.pyplot as plt

def Pacejka_Fx(Fz,slip):
    """
    longitudinal force
    
    :param (float) Fz: Force in vertical direction in N
    :param (float) slip: relative slip fraction (0..1)
    """
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

def Pacejka_Fy(Fz,alpha,camber):
    """
    longitudinal force
    
    :param (float) Fz: Force in vertical direction in N
    :param (float) alpha: slip angle in rad
    :param (float) camber: camber angle in rad
    """
    alpha = alpha * 180.0/np.pi
    camber = camber * 180.0/np.pi
    Fz = Fz/1000.0
    C    = a[0]
    D    = Fz*(a[1]*Fz+a[2])*(1-a[15]*np.power(camber,2))
    BCD  = a[3]*np.sin(np.arctan(Fz/a[4])*2)*(1-a[5]*np.fabs(camber))
    B    = BCD/(C*D)
    H    = a[8]*Fz+a[9]+a[10]*camber
    V    = a[11]*Fz+a[12]+(a[13]*Fz+a[14])*camber*Fz
    E    = (a[6]*Fz+a[7])*(1-(a[16]*camber+a[17])*np.sign(slip+H))
    Bx1  = B*(slip+H)
    Fy   = D*np.sin(C*np.arctan(Bx1-E*(Bx1-np.arctan(Bx1))))+V
    return Fy

Fx_1=[]
Fx_2=[]
Fx_3=[]
Fy_1=[]
Fy_2=[]
Fy_3=[]    
#Fz     = [13.332,26.663,39.995]
#b      = [2.139,0.0045,-0.934,1.971,6.081,0.0654,-0.0014,0.040,2.229,9.716,5.626,3.2e-6,-8.7e-5,0.649]
#a      = [0.384,-0.014,-2.228,241.4,47.72,0,0.642,4.565,12.137,-3.547,0,4.35e-6,2.01e-4,0,0,0,0,0.945]
Fz     = [6,8,10]
b      = [1.5,0.,1100.,0.,300.,0.,0.,0.,-2.,0.,0.,0.,0.,0.]
a      = [1.4,0.,1100.,1100.,10.,0.,0.,-2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]

slip_angle   = np.arange(-30.,30.,0.5)
slip = np.arange(-90.,90.,1.)
camber = 0
for i in Fz:
    for j in slip:
        if i==Fz[0]:
            #print(i)
            Fx_1.append(Pacejka_Fx(i,j))
        elif i==Fz[1]:
            #print i,'---------'
            Fx_2.append(Pacejka_Fx(i,j))
        elif i==Fz[2]:
            #print i,'-----------------------'
            Fx_3.append(Pacejka_Fx(i,j))

for i in Fz:
    for j in slip_angle:
        if i==Fz[0]:
            #print(i)
            Fy_1.append(Pacejka_Fy(i,j,camber))
        elif i==Fz[1]:
            #print i,'---------'
            Fy_2.append(Pacejka_Fy(i,j,camber))
        elif i==Fz[2]:
            #print i,'-----------------------'
            Fy_3.append(Pacejka_Fy(i,j,camber))

plt.plot(slip_angle, Fy_1, 'r-', slip_angle, Fy_2, 'b-', slip_angle, Fy_3, 'g-')

plt.show()
