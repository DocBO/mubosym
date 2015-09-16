# -*- coding: utf-8 -*-
"""
Helper Code to convert strings in symbolic expression and vv
============================================================

Created on Mon Jul 20 17:52:06 2015

@author: oliver
"""
from sympy import symbols, sin, cos, tan, Matrix, sqrt, sign, re
from sympy.physics.mechanics import dynamicsymbols
from numpy import sqrt as np_sqrt


class dummyWorld():
    def __init__(self):
        self.M = None
        self.F = None
        self.dynamic = None
        self.name = None

def str_to_expression(_inp, _str_symbols):
    X = symbols('X')
    a0 = """%s = symbols('%s')"""
    b0 = a0 % (_str_symbols, _str_symbols)
    a1 = """X = %s"""
    b1 = a1 % _inp
    exec(b0)
    exec(b1)
    return X
    
def str_to_dynexpr(_inp, _str_symbols):
    X = symbols('X')
    a0 = """%s = dynamicsymbols('%s')"""
    t = symbols('t')
    b0 = a0 % (_str_symbols, _str_symbols)
    a1 = """X = %s"""
    b1 = a1 % _inp
    exec(b0)
    exec(b1)
    return X
    
def exlist_to_str(_l):
    out_l = []
    for i in  range(len(_l)):
        s = str(_l[i]).replace("(t)","")
        out_l.append(s)
    return out_l

def str_to_exlist(_str_l, _symbols):
    if not type(_symbols) == str:
        _str_symbols = str(_symbols).replace("[","").replace("(t)","").replace("]","").replace(", t", "")
    else:
        _str_symbols = _symbols.replace("[","").replace("(t)","").replace("]","").replace(", t", "")
    l = []
    for i in range(len(_str_l)):
        l.append(str_to_dynexpr(_str_l[i], _str_symbols))
    return l
    
    
def matrix_to_str(_ma):
    rows = _ma.shape[0]
    cols = _ma.shape[1]
    l = []
    for r in range(rows):
        for c in range(cols):
            s = str(_ma[r,c]).replace("(t)", "")
            l.append(s)
    return l

def str_to_matrix(_l, _symbols): #_sym_symbols are eg. myMBS.dynamics
    if not type(_symbols) == str:
        _str_symbols = str(_symbols).replace("[","").replace("(t)","").replace("]","").replace(", t", "")
    else:
        _str_symbols = _symbols.replace("[","").replace("(t)","").replace("]","").replace(", t", "")
    rows = cols = np_sqrt(len(_l))
    m = Matrix.zeros(rows,cols)
    for i in range(len(_l)):
        c = int(i%cols)
        r = int(i/cols)
        m[r,c] = str_to_dynexpr(_l[i], _str_symbols)
    return m
    
def str_to_vec(_l, _symbols): #_sym_symbols are eg. myMBS.dynamics
    if not type(_symbols) == str:
        _str_symbols = str(_symbols).replace("[","").replace("(t)","").replace("]","").replace(", t", "")
    else:
        _str_symbols = _symbols.replace("[","").replace("(t)","").replace("]","").replace(", t", "")
    rows = len(_l)
    cols = 1
    m = Matrix.zeros(rows,cols)
    for i in range(len(_l)):
        m[i,0] = str_to_dynexpr(_l[i], _str_symbols)
    return m
        
def world_to_list(_myMBS):
    l = [_myMBS.name]
    l.append(str(_myMBS.dynamic))
    l.append(matrix_to_str(_myMBS.M))
    l.append(matrix_to_str(_myMBS.F))
    l.append(exlist_to_str(_myMBS.dynamic))
    return l
    
def list_to_world(_l ):
    _myMBS = dummyWorld()
    _myMBS.name = _l[0]
    dyn = _l[1]
    _myMBS.M = str_to_matrix(_l[2], dyn)
    _myMBS.F = str_to_vec(_l[3], dyn)
    _myMBS.dynamic = str_to_exlist(_l[4], dyn)
    t = symbols('t')
    idx = _myMBS.dynamic.index(t)
    kin = _myMBS.dynamic[0:idx]
    kindiff_dict = {}
    for i in range(idx/2):
        kindiff_dict.update({kin[i].diff(t): kin[i+idx/2]})
    _myMBS.kindiff_dict = kindiff_dict
    return _myMBS
    
class worldData():
    def __init__(self, world = ''):
        if world:
            self.l = world_to_list(world)
        else:
            self.l = []

    def get_list(self):
        return self.l
        
    def get_str(self):
        part1 = self.l[:2]
        part2 = self.l[2]
        part3 = self.l[3]
        part4 = self.l[4]
        glue = '...'.join(part1) + '&&' + '...'.join(part2) + '&&' + '...'.join(part3) + '&&' + '...'.join(part4)
        return glue
        
    def put_str(self, inp):
        parts = inp.split('&&')
        parts_a = parts[0].split('...')
        self.l += parts_a
        parts_b = parts[1].split('...')
        self.l.append(parts_b)
        parts_c = parts[2].split('...')
        self.l.append(parts_c)
        parts_d = parts[3].split('...')
        self.l.append(parts_d)
#        print parts_a
#        print "+++++"
#        print parts_b
#        print "+++++"
#        print parts_c
#        print "+++++"
#        print parts_d
#        print "----------"
#        print self.l
    
if __name__ == "__main__":
    u = str_to_expression('sin(x)+2*cos(z)', 'x, z')
    
