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