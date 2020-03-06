# -*- coding: utf-8 -*-
"""
Created on Sat May 27 11:46:09 2017

@author: MZanin
"""

import numpy as np



def GetExpFit( dDistr, infLimit, supLimit, alpha = 0.05 ):

    from scipy.stats.distributions import  t
    from scipy.optimize import curve_fit
        
    dDistr_Log = np.zeros( (supLimit - infLimit, 2) )
    for k in range( supLimit - infLimit ):
        dDistr_Log[ k, 0 ] = k + infLimit
        dDistr_Log[ k, 1 ] = dDistr[ k + infLimit ]
        
    
    def exponenial_func(x, a, b, c):
        return a * np.power(x, -b) + c

    popt, pcov = curve_fit(exponenial_func, dDistr_Log[:, 0], dDistr_Log[:, 1], p0=(1.0, 1.0, 1.0))
    
    perr = np.sqrt(np.diag(pcov))
    
    n = supLimit - infLimit    # number of data points
    p = 3 # number of parameters
    dof = max(0, n - p) # number of degrees of freedom
    
    tval = t.ppf(1.0-alpha/2., dof)
    
    Interval = perr[1]
    corrInterval = perr[1] * tval
    Exp = popt[1]

    return Exp, Interval, corrInterval



def CompareTwoExpFits( Exp1, Interval1, Exp2, Interval2 ):
    
    import scipy 
    
    newExp = Exp1 - Exp2
    newInterval = Interval1 + Interval2
    
    z_score = newExp / newInterval
    p_value = scipy.stats.norm.sf(abs(z_score))*2
    
    return p_value

