# -*- coding: utf-8 -*-
"""
Joint related code
==================
Created on Mon Sep  7 10:51:38 2015

@author: oliver
"""
from sympy import symbols

class MBSjoint(object):
    """
    Class representing a joint.
    """
    def __init__(self, name):
        self.name = name
        self.x, self.y, self.z = symbols('x y z')
        self.phi, self.theta, self.psi = symbols('phi theta psi')
        self.rot_order = [self.phi, self.theta, self.psi]
        self.trans = [self.x, self.y, self.z]
        self.rot_frame = 0
        self.trans_frame = 0
        self.free_list = []
        self.const_list = []
        self.correspondence = {self.phi: 'X', self.theta: 'Y', self.psi: 'Z'}
        self.c_string = 'XYZ'
        self.n_free = 0
    def define_rot_order(self, order):
        self.rot_order = order
        self.c_string = ''
        for s in self.rot_order:
            self.c_string += self.correspondence[s]
    def define_freedoms(self, free_list):
        self.free_list = free_list
        self.n_free = len(free_list)
    def define_constants(self, const_list):
        self.const_list = const_list
        
##########################################################################
# define useful joints here ...       
joints = []

joints.append(MBSjoint('rod-1-cardanic-efficient'))
joints[-1].define_freedoms([joints[-1].psi])
joints[-1].define_constants([joints[-1].y, joints[-1].theta])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 0

joints.append(MBSjoint('rod-1-cardanic'))
joints[-1].define_freedoms([joints[-1].psi])
joints[-1].define_constants([joints[-1].y, joints[-1].theta])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('x-axes'))
joints[-1].define_freedoms([joints[-1].x])
joints[-1].define_constants([])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('y-axes'))
joints[-1].define_freedoms([joints[-1].y])
joints[-1].define_constants([])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('z-axes'))
joints[-1].define_freedoms([joints[-1].z])
joints[-1].define_constants([])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('angle-rod'))
joints[-1].define_freedoms([joints[-1].theta])
joints[-1].define_constants([joints[-1].phi, joints[-1].y])
joints[-1].define_rot_order([joints[-1].theta, joints[-1].phi, joints[-1].psi])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('rod-zero-X'))
joints[-1].define_freedoms([])
joints[-1].define_constants([joints[-1].x])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('rod-zero-Y'))
joints[-1].define_freedoms([])
joints[-1].define_constants([joints[-1].y])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('rod-zero-Z'))
joints[-1].define_freedoms([])
joints[-1].define_constants([joints[-1].z])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('rod-1-revolute'))
joints[-1].define_freedoms([joints[-1].theta])
joints[-1].define_rot_order([joints[-1].theta, joints[-1].phi, joints[-1].psi])
joints[-1].define_constants([joints[-1].y])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('free-3-translate-z-rotate'))
joints[-1].define_freedoms([joints[-1].theta, joints[-1].x, joints[-1].y, joints[-1].z])
joints[-1].define_constants([])
joints[-1].trans_frame = 0
joints[-1].rot_frame = 2

joints.append(MBSjoint('xz-plane'))
joints[-1].define_freedoms([joints[-1].x, joints[-1].z])
joints[-1].define_constants([])
joints[-1].trans_frame = 0
joints[-1].rot_frame = 0

joints.append(MBSjoint('xy-plane'))
joints[-1].define_freedoms([joints[-1].x, joints[-1].y])
joints[-1].define_constants([])
joints[-1].trans_frame = 0
joints[-1].rot_frame = 0

joints.append(MBSjoint('rod-2-cardanic'))
joints[-1].define_freedoms([joints[-1].psi, joints[-1].theta])
joints[-1].define_rot_order([joints[-1].psi, joints[-1].theta, joints[-1].phi])
joints[-1].define_constants([joints[-1].y])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('rod-2-revolute-scharnier'))
joints[-1].define_freedoms([joints[-1].theta, joints[-1].phi])
joints[-1].define_rot_order([joints[-1].theta, joints[-1].phi, joints[-1].psi])
joints[-1].define_constants([joints[-1].y])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('free-3-rotate'))
joints[-1].define_freedoms([joints[-1].phi, joints[-1].theta, joints[-1].psi])
joints[-1].define_constants([])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('free-3-translate'))
joints[-1].define_freedoms([joints[-1].x, joints[-1].y, joints[-1].z])
joints[-1].define_constants([])
joints[-1].trans_frame = 0
joints[-1].rot_frame = 2

joints.append(MBSjoint('revolute-X'))
joints[-1].define_freedoms([joints[-1].phi])
joints[-1].define_constants([])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('revolute-Y'))
joints[-1].define_freedoms([joints[-1].theta])
joints[-1].define_constants([])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('revolute-Z'))
joints[-1].define_freedoms([joints[-1].psi])
joints[-1].define_constants([])
joints[-1].trans_frame = 1
joints[-1].rot_frame = 2

joints.append(MBSjoint('free-6'))
joints[-1].define_freedoms([joints[-1].phi, joints[-1].theta, joints[-1].psi, joints[-1].x, joints[-1].y, joints[-1].z])
joints[-1].define_constants([])
joints[-1].trans_frame = 0
joints[-1].rot_frame = 0

joints_names = [oo.name for oo in joints]
def_joints = dict(zip(joints_names, joints))