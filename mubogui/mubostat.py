# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 08:30:50 2015

widget to display integrator statistics

This should work with PyQt4 and PyQt5, but sadly, installing both packages is not possible under anaconda...
a patch seems to be on the way... 

can be included in an earlier stage as mubovis...
if no vpython is installed, this can be used as a preliminary solution...

"""

#from PyQt5 import QtCore,QtGui

import pyqtgraph as pg

from pyqtgraph import QtGui

___PyQt_VERSION__ = pg.Qt.QtVersion

if ___PyQt_VERSION__.startswith("4"):
    print( "using PyQt4:", ___PyQt_VERSION__ )
elif ___PyQt_VERSION__.startswith("5"):
    print( "using PyQt5:", ___PyQt_VERSION__ )
else:
    assert None    

print( "using pyqtgraph:", pg.__version__ )






#def plotting(self, t_max, dt, plots = 'standard'):
#            #plotting
#    if plots == 'standard':
#        n = len(self.q_flat)
#        n_max = int(t_max/dt)-2
#        plt.subplot(2, 1, 1)
#        lines = plt.plot(self.time[0:n_max], self.x_t[0:n_max, :n])
#        lab = plt.xlabel('Time [sec]')
#        leg = plt.legend(self.dynamic[:n])
#
#        plt.subplot(2, 1, 2)
#        lines = plt.plot(self.time, self.e_kin,self.time,self.e_rot,self.time, self.e_pot,self.time, self.e_tot)
#        lab = plt.xlabel('Time [sec]')
#        leg = plt.legend(['E_kin','E_rot', 'E_pot', 'E_full'])
#        plt.show()
#        #
#    elif plots == 'y-pos':
#        n = len(self.q_flat)
#        n_max = int(t_max/dt)-2
#        plt.subplot(2, 1, 1)
#        lines = plt.plot(self.time[0:n_max], self.state[0:n_max, 4])
#        lab = plt.xlabel('Time [sec]')
#        leg = plt.legend(["y-Pos."])
#
#        plt.subplot(2, 1, 2)
#        lines = plt.plot(self.time, self.e_kin,self.time,self.e_rot,self.time, self.e_pot,self.time, self.e_tot)
#        lab = plt.xlabel('Time [sec]')
#        leg = plt.legend(['E_kin','E_rot', 'E_pot', 'E_full'])
#        plt.show()
#
#    elif plots == 'tire':
#
#        plt.subplot(5, 1, 1)
#        lines = plt.plot(self.time, array(self.model_signals_results[0])[:,0], self.time, array(self.model_signals_results[0])[:,2])
#        #lab = plt.xlabel('Time [sec]')
#        leg1 = plt.legend(['Fx [N]', 'Fz [N]'])
#
#        plt.subplot(5, 1, 2)
#        lines = plt.plot(self.time, array(self.model_signals_results[0])[:,1])
#        #lab = plt.xlabel('Time [sec]')
#        leg1 = plt.legend(['Fy [N]'])
#
#        plt.subplot(5, 1, 3)
#        lines = plt.plot( self.time, array(self.model_signals_results[0])[:,3])
#        #lab = plt.xlabel('Time [sec]')
#        leg2 = plt.legend(['Tz [Nm]'])
#
#        plt.subplot(5, 1, 4)
#        lines = plt.plot( self.time, array(self.model_signals_results[0])[:,4])
#        lab = plt.xlabel('Time [sec]')
#        leg3 = plt.legend(['Slip [%]'])
#
#        plt.subplot(5, 1, 5)
#        lines = plt.plot( self.time, array(self.model_signals_results[0])[:,5])
#        lab = plt.xlabel('Time [sec]')
#        leg4 = plt.legend(['Alpha [grad]'])
#        plt.show()
#
#    elif plots == 'signals':
#        n_signals = len(self.control_signals_obj)
#        for n in range(n_signals):
#            plt.subplot(n_signals, 1, n+1)
#            lines = plt.plot(self.time, array(self.control_signals_results)[:,n])
#            leg = plt.legend(['Signal '+str(n)])
#            lab = plt.xlabel(self.control_signals_obj[n].name+" in "+self.control_signals_obj[n].unit)
#        plt.show()  



















class mbWidget(QtGui.QWidget):
    """
    Define a top-level widget to hold everything
    """
    def __init__(self):
        super(mbWidget,self).__init__()
        
        # create widgets
        self.createWidgets()
        # create GridLayout
        self.createGridLayout()
        # add Widgets to GridLayout
        self.addWidgets()

    def createWidgets(self):
        """
        Create some widgets to be placed inside
        """        
        self.btn = QtGui.QPushButton('press me')
        self.text = QtGui.QLineEdit('enter text')
        self.listw = QtGui.QListWidget()
        self.plot = pg.PlotWidget()
        
    def createGridLayout(self):
        """
        Create a grid layout to manage the widgets size and position
        """        
        
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        
    def addWidgets(self):
        """
        Add widgets to the layout in their proper positions
        """
        self.layout.addWidget(self.btn, 0, 0)   # button goes in upper-left
        self.layout.addWidget(self.text, 1, 0)   # text edit goes in middle-left
        self.layout.addWidget(self.listw, 2, 0)  # list widget goes in bottom-left
        self.layout.addWidget(self.plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows    
        

if __name__ == "__main__":
    
    ## Always start by initializing Qt (only once per application)
    app = QtGui.QApplication([])

    w = mbWidget()

    ## Display the widget as a new window
    w.show()
    
    ## Start the Qt event loop
    app.exec_()