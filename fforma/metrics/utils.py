#!/usr/bin/env python
# coding: utf-8

import numpy as np

def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div
