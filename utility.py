#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:48:36 2017

@author: altescy
"""

import numpy as np


def _clip(x):
	return np.float32(-1 if x< -1 else (1 if x>1 else x))

def clip(x):
     return np.vectorize(_clip)(x)