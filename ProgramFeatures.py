# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:31:40 2017

@author: MZanin
"""


import numpy as np
import networkx as nx
import scipy.io
import glob

import NetworkProperties as netp




for group in [1]:

    allProp = []

    allFiles = glob.glob( './Networks/G%d_*.npy' % group )
    numFiles = len( allFiles )

    for index in range( numFiles ):

        print( 'Processing group %d, network %d' % ( group, index ) )
        fileName = './Networks/G%d_%05d.npy' % ( group, index )
        AMO = np.load( fileName )

        if np.size( AMO, 0 ) > 50 or np.size( AMO, 0 ) < 3:
            continue

        prop = netp.GetAllProperties( AMO )

        rndProp = np.zeros((np.size(prop, 0), 100))
        fileName = './RndNetworks/G%d_%05d.npy' % ( group, index )
        AMR = np.load( fileName )

        for k in range(100):
            AM = AMR[ k, :, : ]
            rndProp[:, k] = netp.GetAllProperties( AM )

        tmpStd = np.std(rndProp, 1)
        normProp = (prop - np.mean(rndProp, 1)) / tmpStd
        normProp[:2] = prop[:2]

        allProp.append( normProp[:] )


    np.save( 'allProp_G%d' % group, allProp )

