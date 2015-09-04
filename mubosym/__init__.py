# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:04:30 2015

@author: oliver
"""
import os

BASE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),"../"))
DATA_PATH = os.path.realpath(os.path.join(BASE_PATH,'data'))

from symTools import matrix_to_str, str_to_matrix, world_to_list, list_to_world
from mubosym_core import MBSworld

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
