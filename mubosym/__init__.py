# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:04:30 2015

@author: oliver
"""
import os, sys

BASE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),"../"))
LOCAL_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),"./"))
DATA_PATH = os.path.realpath(os.path.join(BASE_PATH,'data'))

sys.path.append(BASE_PATH)
sys.path.append(LOCAL_PATH)
sys.path.append(DATA_PATH)


from mubosym_core import MBSworld, def_joints

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
