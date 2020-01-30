# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:48:52 2017

@author: MZanin
"""

import numpy as np
import ExpFit as ef



Exponents = np.zeros( (2, 1) )
Intervals = np.zeros( (2, 1) )

sizeC = np.load('netSizes1.npy')
infLimit = 10
supLimit = 80
Exponents[0], Intervals[0], _ = ef.GetExpFit( sizeC, infLimit, supLimit )

sizeP = np.load('netSizes2.npy')
infLimit = 10
supLimit = 80
Exponents[1], Intervals[1], _ = ef.GetExpFit( sizeP, infLimit, supLimit )



PValues = np.zeros( (2, 2) )
for k1 in range(2):
    for k2 in range(2):
        PValues[k1, k2] = ef.CompareTwoExpFits( Exponents[k1], Intervals[k1], Exponents[k2], Intervals[k2] )

